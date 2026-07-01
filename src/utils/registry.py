"""A tiny string-keyed registry so configs can select components by name.

Usage:
    from src.utils.registry import register, build

    @register("model", "compact3dcnn")
    class Compact3DCNN(nn.Module): ...

    model = build("model", "compact3dcnn", num_classes=27)

Kinds used in this project: "model", "dataset", "peft", "loss", "scheduler".
The registry is deliberatly minimal — no plugin magic — so reproduction is
obvious from reading the code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class _Registry:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Callable]] = {}

    def register(self, kind: str, name: str) -> Callable[[Callable], Callable]:
        def deco(obj: Callable) -> Callable:
            bucket = self._store.setdefault(kind, {})
            if name in bucket:
                raise KeyError(f"{kind!r} already has an entry named {name!r}")
            bucket[name] = obj
            return obj

        return deco

    def get(self, kind: str, name: str) -> Callable:
        try:
            return self._store[kind][name]
        except KeyError as e:
            available = sorted(self._store.get(kind, {}).keys())
            raise KeyError(
                f"No {kind!r} registered as {name!r}. Available: {available}"
            ) from e

    def build(self, kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(kind, name)(*args, **kwargs)

    def available(self, kind: str) -> list[str]:
        return sorted(self._store.get(kind, {}).keys())


REGISTRY = _Registry()
register = REGISTRY.register
build = REGISTRY.build
