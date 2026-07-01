"""Deterministic seeding across python / numpy / torch.

Reproducibility is a hard requirement (BRIEF section 11): we set and *log* the
seed in every results artifact. `seed_everything` is intentionally strict by
default but exposes `deterministic_algorithms` so that throughput benchmarks can
opt out of the (slower) deterministic cuDNN kernels when only timing matters.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic_algorithms: bool = True) -> int:
    """Seed python, numpy and torch RNGs.

    Args:
        seed: the integer seed (logged into every artifact).
        deterministic_algorithms: if True, force deterministic cuDNN/torch
            kernels. Disable for pure timing benchmarks where the small
            nondeterminism is acceptable and the deterministic kernels would
            distort the latency numbers.

    Returns:
        the seed (so callers can log the resolved value).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CUBLAS workspace config is required for full determinism with
            # some matmul kernels; harmless if already set.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # Older torch may not support warn_only; fall back silently.
                pass
        else:
            # Benchmark mode: let cuDNN pick the fastest kernels.
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch not installed yet (e.g. during scaffold-only smoke); fine.
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding so multi-worker loading stays reproducible."""
    base_seed = np.random.get_state()[1][0]
    seed = (int(base_seed) + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
