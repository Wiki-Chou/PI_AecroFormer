"""The runtime policy is process-local: CPU, two threads, one worker."""
from __future__ import annotations

import os


def configure_runtime(threads: int | None = None) -> int:
    count = int(threads if threads is not None else os.environ.get("PI_AECROFORMER_THREADS", "2"))
    if not 1 <= count <= 4:
        raise ValueError("PI_AECROFORMER_THREADS must be between 1 and 4 (default: 2).")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(count)
    # The same miepython kernels are compiled on CPU; no approximation is used.
    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")
    return count


THREADS = configure_runtime()

