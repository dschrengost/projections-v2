"""Process-level runtime knobs to reduce native-library instability.

This project leans on heavy native dependencies (numpy/pandas/pyarrow/BLAS).
When these stacks misbehave, failures present as hard crashes (SIGSEGV/double free)
that bypass normal Python exception handling.

We set conservative defaults early in entrypoints to reduce thread contention and
ensure faulthandler dumps a Python traceback on fatal signals.
"""

from __future__ import annotations

import faulthandler
import os


_DEFAULT_ENV: dict[str, str] = {
    # Dump tracebacks on fatal signals (segfaults, abort, etc).
    "PYTHONFAULTHANDLER": "1",
    # Arrow / BLAS / numeric stack: keep thread counts low to avoid oversubscription
    # and reduce the surface area for native crashes under load.
    "ARROW_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    # glibc allocator: reduce arenas (helps memory fragmentation and can reduce
    # allocator-related crashes in some multi-process workloads).
    "MALLOC_ARENA_MAX": "2",
}


def configure_runtime_safety(*, enable_faulthandler: bool = True) -> None:
    """Apply safe env defaults for this Python process.

    Call this before importing numpy/pandas/pyarrow in entrypoints.
    """

    for key, value in _DEFAULT_ENV.items():
        os.environ.setdefault(key, value)

    if enable_faulthandler:
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            # Best-effort: avoid masking the real workload if faulthandler cannot initialize.
            pass

