"""Hold each vLLM worker's ``init_device`` until the job's startup barrier is released.

See ``inference_servers/startup_barrier.py``. ``init_device`` is where the worker first takes a
CUDA context, checks free memory against ``gpu_memory_utilization`` and snapshots the baseline
that profiling is measured from, so it is the one place the trainer must be offloaded before.
Imported (and thereby applied) by ``NewInferenceWorkerWrap`` so it runs in every worker process.
A worker that finds no barrier in its job proceeds immediately.
"""

from vllm.v1.worker.gpu_worker import Worker

from skyrl.backends.skyrl_train.inference_servers.startup_barrier import (
    wait_for_startup_barrier,
)

_original_init_device = Worker.init_device


def _init_device_after_barrier(self, *args, **kwargs):
    wait_for_startup_barrier()
    return _original_init_device(self, *args, **kwargs)


if not getattr(Worker, "_skyrl_startup_barrier_patched", False):
    Worker.init_device = _init_device_after_barrier
    Worker._skyrl_startup_barrier_patched = True
