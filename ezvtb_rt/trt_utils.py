import os
from pathlib import Path
import time
import numpy as np
import tensorrt_rtx as trt
from typing import List, Dict, Tuple
import pycuda.driver as cuda
from os.path import join
import numpy
from ezvtb_rt.trt_cache import (
    EngineCacheRequiredError,
    atomic_write,
    engine_build_lock,
    get_engine_cache_path as _get_engine_cache_path,
    get_runtime_cache_path as _get_runtime_cache_path,
    load_runtime_cache,
    save_runtime_cache,
)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
GPU_DUTY_LIMIT_ENV = "EZVTB_GPU_DUTY_LIMIT"
REQUIRE_ENGINE_CACHE_ENV = "EZVTB_TRT_REQUIRE_ENGINE_CACHE"

# Solution from https://github.com/NVIDIA/TensorRT/issues/1050#issuecomment-775019583
def cudaSetDevice(device_idx):
    from ctypes import cdll, c_char_p
    libcudart = cdll.LoadLibrary('cudart64_12.dll')
    libcudart.cudaGetErrorString.restype = c_char_p
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + str(error_string))


def get_gpu_duty_limit_percent() -> float:
    """Read the optional startup duty-cycle limit without changing defaults."""
    try:
        limit = float(os.environ.get(GPU_DUTY_LIMIT_ENV, "100"))
    except ValueError:
        return 100.0
    if not 0 < limit <= 100:
        return 100.0
    return limit


def pace_gpu_startup(started_at: float, operation: str) -> float:
    """Add conservative cooldown after an indivisible TensorRT startup call."""
    limit = get_gpu_duty_limit_percent()
    if limit >= 100:
        return 0.0

    active_seconds = max(0.0, time.perf_counter() - started_at)
    cooldown_seconds = active_seconds * (100.0 / limit - 1.0)
    if cooldown_seconds > 0:
        TRT_LOGGER.log(
            TRT_LOGGER.INFO,
            f'GPU safety cooldown after {operation}: {cooldown_seconds:.3f}s '
            f'(target duty cycle {limit:.1f}%)',
        )
        time.sleep(cooldown_seconds)
    return cooldown_seconds


def _trt_cache_identity() -> str:
    device_id_text = os.environ.get("EZVTB_DEVICE_ID", "0")
    identity = [
        getattr(trt, '__version__', 'unknown'),
        f'device-id={device_id_text}',
    ]
    try:
        # Query the configured device directly instead of consulting the active
        # context.  The latter made cache keys depend on import/startup order:
        # calls before ``pycuda.autoinit`` omitted the GPU identity while calls
        # after it included one, causing needless duplicate engine builds.
        cuda.init()
        device = cuda.Device(int(device_id_text))
        identity.append(f'name={device.name()}')
        identity.append(f'cc={device.compute_capability()}')
    except Exception:
        # Cache paths can still be inspected on systems without a CUDA driver.
        # TensorRT deserialization remains the final compatibility guard.
        pass
    return '|'.join(identity)


def get_engine_cache_path(onnx_path: str) -> Path:
    return _get_engine_cache_path(onnx_path, _trt_cache_identity())


def get_runtime_cache_path(source_path: str) -> Path:
    return _get_runtime_cache_path(source_path, _trt_cache_identity())

def build_engine(onnx_file_path:str) -> bytes:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        parse_res = parser.parse(model.read())
        if not parse_res:
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    def GiB(val):
        return val * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(4)) # 4G
    config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL
    config.num_compute_capabilities = 1
    config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)
    
    def is_dynamic_shape()->bool:
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            dims = network.get_input(i).shape
            if dims[0] == -1:
                return True
        return False

    if is_dynamic_shape():
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            print('Setting dynamic shape for input:', input_name)
            dims = network.get_input(i).shape
            min_shape = trt.Dims(dims)
            opt_shape = trt.Dims(dims)
            max_shape = trt.Dims(dims)
            min_shape[0] = 1
            opt_shape[0] = 4
            max_shape[0] = 4
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            TRT_LOGGER.log(TRT_LOGGER.INFO, f'Setting dynamic shape for input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}')
        config.add_optimization_profile(profile)
        config.builder_optimization_level = 5
    # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    build_started_at = time.perf_counter()
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    finally:
        pace_gpu_startup(build_started_at, f'engine build ({Path(onnx_file_path).name})')
    if serialized_engine is None:
        raise RuntimeError(f'Failed to build TensorRT engine from {onnx_file_path}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    atomic_write(path, engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')


def _deserialize_engine(path: Path):
    """Validate and deserialize one engine, returning None when it is unusable."""
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        header_size = getattr(runtime, 'engine_header_size', None)
        if hasattr(runtime, 'get_engine_validity') and header_size:
            with open(path, 'rb') as engine_file:
                header = engine_file.read(header_size)
            if len(header) != header_size:
                TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Engine cache is truncated: {path}')
                return None
            validity, diagnostics = runtime.get_engine_validity(memoryview(header))
            invalid = getattr(getattr(trt, 'EngineValidity', None), 'INVALID', None)
            if invalid is not None and validity == invalid:
                TRT_LOGGER.log(
                    TRT_LOGGER.WARNING,
                    f'Engine cache is incompatible (diagnostics={diagnostics}): {path}',
                )
                return None

        with open(path, 'rb') as engine_file:
            engine = runtime.deserialize_cuda_engine(engine_file.read())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Unable to load engine cache {path}: {error}')
        return None

    if engine is None:
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Engine deserialization failed: {path}')
    return engine

def load_engine(path):
    source_path = Path(path)
    if source_path.suffix.lower() != '.onnx':
        TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {source_path}')
        engine = _deserialize_engine(source_path)
        if engine is None:
            raise RuntimeError(f'Failed to load TensorRT engine: {source_path}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
        return engine

    engine_path = get_engine_cache_path(str(source_path))
    if engine_path.is_file():
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading cached engine: {engine_path}')
        engine = _deserialize_engine(engine_path)
        if engine is not None:
            return engine

    if os.environ.get(REQUIRE_ENGINE_CACHE_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        raise EngineCacheRequiredError(
            f'An existing valid TensorRT engine is required: {engine_path}. '
            'Engine building is disabled during guarded runtime-cache startup/recovery.'
        )

    # Recheck after taking the lock: another EasyVtuber process may have built it.
    with engine_build_lock(engine_path):
        if engine_path.is_file():
            engine = _deserialize_engine(engine_path)
            if engine is not None:
                return engine

        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building engine from ONNX: {source_path}')
        serialized_engine = build_engine(str(source_path))
        save_engine(serialized_engine, engine_path)
        engine = _deserialize_engine(engine_path)
        if engine is None:
            raise RuntimeError(f'Newly built TensorRT engine could not be loaded: {engine_path}')
        return engine
