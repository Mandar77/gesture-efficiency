"""Model definitions. Importing this package populates the model registry.

Models register themselves via `@register("model", "<name>")`. Import them here
so `build("model", name, ...)` works after `import src.models`.
"""

from src.models import compact3dcnn  # noqa: F401  (compact3dcnn + dummy)

# Foundation-model teacher, streaming student, and fusion heads are imported
# lazily by their training entrypoints to avoid pulling timm/open_clip at
# scaffold time. They register on import of their modules.
_OPTIONAL = ["student", "peft_teacher", "fusion"]
for _m in _OPTIONAL:
    try:  # pragma: no cover - optional heavy deps
        __import__(f"src.models.{_m}")
    except Exception:
        pass
