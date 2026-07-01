"""Shared utilities: seeding, logging, config loading, registry, checkpoint IO,
and environment metadata capture.

These define the stable contracts the rest of the codebase builds against:
    - `seed_everything(seed)`            -> deterministic runs
    - `get_logger(name)`                 -> stdlib logging, consistent format
    - `load_config(path, overrides)`     -> dict from YAML + CLI dotted overrides
    - `REGISTRY` / `register(kind, name)`-> build models/datasets/etc by name
    - `env_metadata()`                   -> dict of GPU/CUDA/torch/seed/timestamp
    - `save_checkpoint` / `load_checkpoint`
"""

from src.utils.seeding import seed_everything, worker_init_fn
from src.utils.logging_utils import get_logger, setup_file_logging
from src.utils.config import load_config, save_config, dict_to_namespace
from src.utils.registry import REGISTRY, register, build
from src.utils.env import env_metadata, count_parameters
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.results import ResultsWriter

__all__ = [
    "seed_everything",
    "worker_init_fn",
    "get_logger",
    "setup_file_logging",
    "load_config",
    "save_config",
    "dict_to_namespace",
    "REGISTRY",
    "register",
    "build",
    "env_metadata",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "ResultsWriter",
]
