import time
from contextlib import contextmanager
import pycuda.driver as cuda

@contextmanager
def latency_timer(metrics: dict):
    metrics.setdefault("t_latency", 0)
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    metrics["t_latency"] += (end - start)

@contextmanager
def kernel_timer(metrics: dict):
    metrics.setdefault("t_kernel", 0)
    start, end = cuda.Event(), cuda.Event()
    start.record()
    yield
    end.record()
    end.synchronize()
    metrics["t_kernel"] += start.time_till(end) * 1e-3