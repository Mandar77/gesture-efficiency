"""Unit tests for the model zoo: PEFT teacher, streaming student, multimodal
fusion. CPU-friendly, tiny configs. Verifies the shared contracts:
    forward([B,C,T,H,W]) -> [B,num_classes]; return_features -> (logits, feats);
    feature_dim exposed; PEFT trainable < total.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

import pytest
import torch

warnings.filterwarnings("ignore")

import src.models  # noqa: F401
from src.utils.env import count_parameters
from src.utils.registry import build


def test_streaming_student_forward_and_params():
    m = build("model", "streaming_student", num_classes=10)
    x = torch.randn(2, 3, 8, 64, 64)
    logits = m(x)
    assert logits.shape == (2, 10)
    logits, feats = m(x, return_features=True)
    assert feats.shape[0] == 2 and feats.shape[1] == m.feature_dim
    n = count_parameters(m)["total"]
    # Documented ~3.1M at defaults; allow a broad band so refactors don't break CI.
    assert 1_000_000 < n < 8_000_000, f"unexpected param count {n}"


def test_streaming_student_stream_api():
    m = build("model", "streaming_student", num_classes=7).eval()
    m.reset_stream()
    out = None
    for _ in range(8):
        frame = torch.randn(2, 3, 64, 64)
        out = m.forward_step(frame)
    assert out.shape == (2, 7)


def test_streaming_matches_batch_in_eval():
    """Streaming last-step should closely match whole-clip forward in eval mode."""
    torch.manual_seed(0)
    m = build("model", "streaming_student", num_classes=5, width=16, blocks=(1, 1)).eval()
    clip = torch.randn(1, 3, 6, 32, 32)
    with torch.no_grad():
        batch_logits = m(clip)
        m.reset_stream()
        step_logits = None
        for t in range(clip.shape[2]):
            step_logits = m.forward_step(clip[:, :, t])
    # Causal running-avg pooling makes the last step comparable to the clip mean.
    assert torch.allclose(batch_logits, step_logits, atol=1e-3)


@pytest.mark.parametrize("fusion", ["late_logit", "late_feature", "shared_adapter"])
def test_multimodal_fusion_forward(fusion):
    m = build("model", "multimodal_fusion", num_classes=8,
              modalities=("rgb", "depth", "ir"), fusion=fusion,
              encoder_kwargs={"width": 8, "depth": 2})
    batch = {k: torch.randn(2, 3, 8, 32, 32) for k in ("rgb", "depth", "ir")}
    logits = m(batch)
    assert logits.shape == (2, 8)
    # RGB-only subset still runs (mask-based ablation).
    rgb_only = {"rgb": batch["rgb"]}
    assert m(rgb_only).shape == (2, 8)
    logits, feats = m(batch, return_features=True)
    assert feats.shape[0] == 2


def test_shared_adapter_is_cheaper_than_late_feature():
    common = dict(num_classes=8, modalities=("rgb", "depth", "ir"),
                  encoder_kwargs={"width": 8, "depth": 2})
    late = build("model", "multimodal_fusion", fusion="late_feature", **common)
    shared = build("model", "multimodal_fusion", fusion="shared_adapter", **common)
    n_late = count_parameters(late)["total"]
    n_shared = count_parameters(shared)["total"]
    # Shared backbone => far fewer params as modalities grow (efficiency angle).
    assert n_shared < n_late


def test_peft_teacher_constructs_and_trainable_fraction():
    # Use pretrained=False path implicitly (may download; if offline, timm falls
    # back). Small ViT + patch16 so 64px divides into a patch grid.
    pytest.importorskip("timm")
    for method in ("none", "lora", "adapter", "prompt"):
        m = build("model", "peft_teacher", num_classes=6,
                  backbone="vit_small_patch16_224", peft_method=method,
                  temporal_layers=1, temporal_heads=6)
        x = torch.randn(2, 3, 4, 64, 64)
        logits = m(x)
        assert logits.shape == (2, 6)
        p = count_parameters(m)
        # For PEFT (not full_ft), trainable must be a strict subset of total.
        assert p["trainable"] < p["total"], f"{method}: trainable !< total"
        assert m.feature_dim == m.embed_dim
