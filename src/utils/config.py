"""YAML config loading with dotted-key CLI overrides and `_base_` inheritance.

Config-driven everything (BRIEF section 7): no magic numbers in code, seeds in
config. A config may declare `_base_: path/to/parent.yaml` to inherit and
override; CLI overrides are applied last as `--set a.b.c=value` style dotted
keys (parsed by the caller into a list of "a.b.c=value" strings).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    """Best-effort parse a CLI string into int/float/bool/list/None/str."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_override(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override into non-dict at {k!r} ({dotted_key})")
    node[keys[-1]] = value


def load_config(
    path: str | Path,
    overrides: Optional[List[str]] = None,
    _seen: Optional[set] = None,
) -> Dict[str, Any]:
    """Load a YAML config, resolving `_base_` inheritance and CLI overrides.

    Args:
        path: path to the YAML file.
        overrides: list of "dotted.key=value" strings applied after loading.
        _seen: internal guard against circular `_base_` references.
    """
    path = Path(path)
    _seen = _seen or set()
    rp = path.resolve()
    if rp in _seen:
        raise ValueError(f"Circular _base_ reference at {rp}")
    _seen.add(rp)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path, overrides=None, _seen=_seen)
        cfg = _deep_merge(base_cfg, cfg)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} is not of the form key=value")
        key, raw = ov.split("=", 1)
        _apply_override(cfg, key.strip(), _coerce(raw.strip()))

    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Dump a resolved config next to its run artifacts for reproduction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a dict to a namespace for dotted attribute access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns
