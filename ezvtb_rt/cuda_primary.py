"""Initialize the selected device's primary CUDA context for the TRT worker.

Like pycuda.autoprimaryctx, this module owns one context-stack entry until
process exit. It selects EZVTB_DEVICE_ID explicitly instead of letting
CUDA_DEVICE or .cuda_device silently choose a different GPU.
"""

import atexit
import os

import pycuda.driver as cuda


cuda.init()
device = cuda.Device(int(os.environ.get("EZVTB_DEVICE_ID", "0")))
context = device.retain_primary_context()
try:
    context.push()
except Exception:
    context.detach()
    raise


def _finish_up():
    global context
    if context is None:
        return
    retained_context = context
    retained_context.pop()
    context = None
    try:
        from pycuda.tools import clear_context_caches

        clear_context_caches()
    finally:
        retained_context.detach()


atexit.register(_finish_up)
