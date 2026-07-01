"""Dataset definitions + a `build_dataloaders` helper.

Datasets register via `@register("dataset", "<name>")`. Importing this package
registers the synthetic dataset always; real datasets (jester/nvgesture/shrec)
register on import of their modules.
"""

from src.data import synthetic  # noqa: F401  (always available for smoke)

for _m in ["jester", "nvgesture", "shrec"]:
    try:
        __import__(f"src.data.{_m}")
    except Exception:
        pass

from src.data.loaders import build_dataloaders  # noqa: E402

__all__ = ["build_dataloaders"]
