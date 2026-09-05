"""TensorRT engine management module for real-time inference operations.

This module handles:
- TensorRT engine initialization
- Memory allocation for input/output tensors
- Execution context management
- Asynchronous inference operations
"""

from ezvtb_rt.trt_utils import *
import os
from pathlib import Path
import time


PROFILE_INFERENCE_ENV = 'EZVTB_TRT_PROFILE'
RUNTIME_CACHE_ENV = 'EZVTB_TRT_RUNTIME_CACHE'


def _profile_inference_enabled() -> bool:
    return os.environ.get(PROFILE_INFERENCE_ENV, '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _runtime_cache_enabled() -> bool:
    """Opt in while TensorRT-RTX 1.3 runtime-cache I/O is unstable."""
    return os.environ.get(RUNTIME_CACHE_ENV, '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )

#memory management
class HostDeviceMem(object):
    def __init__(self, host_mem:numpy.ndarray, device_mem: cuda.DeviceAllocation):
        self.host: numpy.ndarray = host_mem
        self.device: cuda.DeviceAllocation = device_mem

    @classmethod
    def create(cls, shape, dtype):
        host_mem = cuda.pagelocked_empty(shape, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return cls(host_mem, device_mem)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    def __del__(self):
        self.device.free()

    def dtoh(self, stream:cuda.Stream):
        cuda.memcpy_dtoh_async(self.host, self.device, stream) 
    def htod(self, stream:cuda.Stream):
        cuda.memcpy_htod_async(self.device, self.host, stream)

    def bridgeFrom(self, other: 'HostDeviceMem', stream:cuda.Stream):
        assert self.host.nbytes == other.host.nbytes, f"Memory sizes must match for bridging, dst: {self.host.nbytes}, src: {other.host.nbytes}, dst shape: {self.host.shape}, src shape: {other.host.shape}"
        cuda.memcpy_dtod_async(self.device, other.device, self.host.nbytes, stream)



class TRTEngine:
    def __init__(
            self,
            engine: trt.ICudaEngine | str,
            n_input:int,
            profile_inference: bool | None = None,
    ):
        source_path = os.fspath(engine) if isinstance(engine, (str, os.PathLike)) else None
        if source_path is not None:
            engine = load_engine(source_path)
            assert engine is not None, f'Failed to load engine from path {engine}'
        self.engine: trt.ICudaEngine = engine
        source_name = Path(source_path).name if source_path else 'in-memory engine'
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Creating inference context: {source_name}')
        # create execution context
        self.runtime_config = engine.create_runtime_config()
        self.runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        self.runtime_cache = None
        self.runtime_cache_path = None
        if source_path is not None and _runtime_cache_enabled():
            try:
                self.runtime_cache_path = get_runtime_cache_path(source_path)
                self.runtime_cache, cache_loaded = load_runtime_cache(
                    self.runtime_config,
                    self.runtime_cache_path,
                )
                if cache_loaded:
                    TRT_LOGGER.log(
                        TRT_LOGGER.INFO,
                        f'Loaded TensorRT runtime cache: {self.runtime_cache_path}',
                    )
            except Exception as error:
                self.runtime_cache = None
                TRT_LOGGER.log(
                    TRT_LOGGER.WARNING,
                    f'Runtime cache unavailable for {source_path}: {error}',
                )

        context_started_at = time.perf_counter()
        try:
            self.context: trt.IExecutionContext = engine.create_execution_context(
                self.runtime_config,
            )
        finally:
            source_name = Path(source_path).name if source_path else 'in-memory engine'
            pace_gpu_startup(
                context_started_at,
                f'context creation ({source_name})',
            )
        if self.context is None:
            raise RuntimeError('TensorRT failed to create an inference context')

        # Keep the runtime config/cache alive for the lifetime of the context.
        # Persist after successful inference and stream completion, including
        # for static models. Dynamic background specialization may require
        # additional handling; stream completion alone does not wait for it.
        self._runtime_cache_save_pending = self.runtime_cache is not None
        self.n_batch: int = -1
        self.in_out_tensors: dict = {}
        self.inputs: List[HostDeviceMem] = []
        self.outputs: List[HostDeviceMem] = []
        
        # get input and output tensor names
        self.input_tensor_names: List[str] = [engine.get_tensor_name(i) for i in range(n_input)]
        self.output_tensor_names: List[str] = [engine.get_tensor_name(i) for i in range(n_input, self.engine.num_io_tensors)]
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Input nodes: '+ str(self.input_tensor_names))
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Output nodes: '+ str(self.output_tensor_names))

        self.input_tensors_original_shapes: Dict[str, List[int]] = {}
        self.output_tensors_original_shapes: Dict[str, List[int]] = {}
        for input_tensor_name in self.input_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(input_tensor_name)]
            self.input_tensors_original_shapes[input_tensor_name] = shape
        for output_tensor_name in self.output_tensor_names:
            shape = [dim for dim in self.context.get_tensor_shape(output_tensor_name)]
            self.output_tensors_original_shapes[output_tensor_name] = shape

        # create stream
        self.stream: cuda.Stream = cuda.Stream()
        # CUDA timing events add work to every enqueue. Keep them disabled in
        # normal operation and create them only for an explicit profiling run.
        if profile_inference is None:
            profile_inference = _profile_inference_enabled()
        self.profile_inference = bool(profile_inference)
        self.start_event: cuda.Event | None = (
            cuda.Event() if self.profile_inference else None
        )
        self.end_event: cuda.Event | None = (
            cuda.Event() if self.profile_inference else None
        )
            
    def get_last_inference_time(self):
        if self.start_event is None or self.end_event is None:
            raise RuntimeError(
                'TensorRT inference profiling is disabled; pass '
                'profile_inference=True or set EZVTB_TRT_PROFILE=1 before '
                'creating the engine'
            )
        return self.start_event.time_till(self.end_event)

    def _persist_runtime_cache(self):
        if self.runtime_cache is None or self.runtime_cache_path is None:
            return False
        try:
            save_runtime_cache(self.runtime_cache, self.runtime_cache_path)
        except Exception as error:
            TRT_LOGGER.log(
                TRT_LOGGER.WARNING,
                f'Unable to save TensorRT runtime cache {self.runtime_cache_path}: {error}',
            )
            return False
        return True

    def putInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch: int = 1, stream: cuda.Stream = None, sync: bool = False):
        """Non-blocking input upload - doesn't synchronize stream"""
        stream = stream if stream is not None else self.stream
        input_tensors, output_tensors = self.configure_in_out_tensors(n_batch)
        for inp, inp_mem, inp_name in zip(np_inputs, input_tensors, self.input_tensor_names):
            if inp.dtype != inp_mem.host.dtype or inp.shape != inp_mem.host.shape:
                print('Given:', inp.dtype, inp.shape)
                print('Expected:', inp_mem.host.dtype, inp_mem.host.shape)
                raise ValueError(f'Input shape or type does not match for input tensor {inp_name}')
            if isinstance(inp, HostDeviceMem):
                cuda.memcpy_dtod_async(inp_mem.device, inp.device, 
                                   inp_mem.host.nbytes, stream)
            else:
                np.copyto(inp_mem.host, inp)
                inp_mem.htod(stream)
        if sync:
            stream.synchronize()

    def syncPutInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch:int = 1, stream: cuda.Stream = None):
        self.putInputs(np_inputs, n_batch, stream, sync=True)
    
    def asyncPutInputs(self, np_inputs: List[np.ndarray | HostDeviceMem], n_batch:int = 1, stream: cuda.Stream = None):
        self.putInputs(np_inputs, n_batch, stream, sync=False)

    def kickoff(self, stream: cuda.Stream = None, sync:bool = False):
        if stream is None:
            stream = self.stream
        if self.start_event is not None:
            self.start_event.record(stream)
        # Run inference.
        if not self.context.execute_async_v3(stream.handle):
            raise RuntimeError('TensorRT inference enqueue failed; runtime cache was not saved')
        if self.end_event is not None:
            self.end_event.record(stream)

        # Runtime-cache data may be populated by the first shape-specific
        # enqueue. NVIDIA's documented flow saves it only after inference has
        # completed. Serializing here before the asynchronous stream finished
        # could produce a blob that deserialized successfully but stalled the
        # next process during execution-context creation. Pay one synchronization
        # per newly configured batch shape, then persist a complete cache.
        cache_save_pending = self._runtime_cache_save_pending
        if sync or cache_save_pending:
            stream.synchronize()
        if cache_save_pending:
            self._persist_runtime_cache()
            self._runtime_cache_save_pending = False

    def asyncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=False)

    def syncKickoff(self, stream: cuda.Stream = None):
        self.kickoff(stream, sync=True)

    # This is a numpy cpu interface for convenience
    def syncGetOutputs(self, copy:bool = True) -> List[np.ndarray]:
        for out_mem in self.outputs: out_mem.dtoh(self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        if copy:
            return [np.copy(outp.host) for outp in self.outputs]
        else:
            return [outp.host for outp in self.outputs]
    
    # This is a numpy cpu interface for convenience
    def syncInfer(self, np_inputs: List[np.ndarray], n_batch:int = 1) -> List[np.ndarray]:
        self.syncPutInputs(np_inputs, n_batch)
        self.asyncKickoff()
        return self.syncGetOutputs(True)
    
    def configure_in_out_tensors(self, n_batch:int = 1) -> Tuple[List[HostDeviceMem], List[HostDeviceMem]]:
        inputs: List[HostDeviceMem] = []
        outputs: List[HostDeviceMem] = []
        if  n_batch in self.in_out_tensors:
            inputs, outputs = self.in_out_tensors[n_batch]
            if n_batch != self.n_batch:
                # TRT_LOGGER.log(TRT_LOGGER.INFO, f'Reconfiguring tensor addresses for batch size {n_batch}')
                for input_tensor_name, input_mem in zip(self.input_tensor_names, inputs):
                    self.context.set_tensor_address(input_tensor_name, int(input_mem.device))
                    self.context.set_input_shape(input_tensor_name, trt.Dims(input_mem.host.shape))
                for output_tensor_name, output_mem in zip(self.output_tensor_names, outputs):
                    self.context.set_tensor_address(output_tensor_name, int(output_mem.device))
        else:
            self._runtime_cache_save_pending = self.runtime_cache is not None
            for input_tensor_name in self.input_tensor_names:
                shape = self.input_tensors_original_shapes[input_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(input_tensor_name))
                # TRT_LOGGER.log(TRT_LOGGER.INFO, f'Allocating input tensor: {input_tensor_name} with shape {shape} and dtype {dtype}')
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(input_tensor_name, int(mem.device)) # Use this setup without binding for v3
                self.context.set_input_shape(input_tensor_name, trt.Dims(shape))
                inputs.append(mem)
            for output_tensor_name in self.output_tensor_names:
                shape = self.output_tensors_original_shapes[output_tensor_name].copy()
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = n_batch
                dtype = trt.nptype(self.engine.get_tensor_dtype(output_tensor_name))
                mem = HostDeviceMem.create(shape, dtype)
                self.context.set_tensor_address(output_tensor_name, int(mem.device)) # Use this setup without binding for v3
                outputs.append(mem)
            self.in_out_tensors[n_batch] = (inputs, outputs)
        self.n_batch = n_batch
        self.inputs = inputs
        self.outputs = outputs
        return inputs, outputs
