"""Verification gate for the PEFT sweep (pre-M5 audit).

For each arm (adapter, prompt, full_ft, lora, none) built from the SAME config
that the sweep uses, print:
  - trainable / total params and trainable % (backbone-only and overall);
  - the exact clip length, spatial resolution, batch size (from the config);
  - peak VRAM measured with torch.cuda.max_memory_allocated() AFTER a real
    train step (forward + backward + optimizer step), with reset at step start;
  - confirmation full_ft has ~100% trainable (backbone genuinely unfrozen);
  - whether any teacher-feature cache is involved (it is not — the teacher does
    a live ViT forward/backward here; there is no precompute-cache code path).

All arms are instantiated with identical clip/res/batch so the only thing that
varies is what is trainable — that is the validity condition for the PEFT-vs-
full-FT comparison. Prints a compact table; the caller records it in SANITY.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.models  # noqa: F401
from src.utils import load_config
from src.utils.env import count_parameters
from src.utils.registry import build


def measure_arm(method: str, cfg: dict, device: torch.device) -> dict:
    dcfg, tcfg = cfg["data"], cfg["train"]
    frames, size, bs = dcfg["num_frames"], dcfg["frame_size"], tcfg["batch_size"]
    mkwargs = dict(cfg["model"].get("kwargs", {}))
    mkwargs.update(
        num_classes=dcfg["num_classes"], peft_method=method,
        lora_rank=cfg["peft"]["lora_rank"], lora_alpha=cfg["peft"]["lora_alpha"],
        lora_targets=cfg["peft"]["lora_targets"], adapter_dim=cfg["peft"]["adapter_dim"],
        prompt_tokens=cfg["peft"]["prompt_tokens"], frame_size=size,
        grad_checkpointing=cfg.get("grad_checkpointing", True),
    )
    model = build("model", "peft_teacher", **mkwargs).to(device)

    overall = count_parameters(model)
    bk = count_parameters(model.backbone)

    # Real train step: reset peak, forward+backward+step, then read peak.
    peak_mb = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, frames, size, size, device=device)
    y = torch.randint(0, dcfg["num_classes"], (bs,), device=device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        loss = ce(model(x), y)
    loss.backward()          # measure AFTER backward (activations + grads live)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "method": method,
        "frames": frames, "resolution": size, "batch_size": bs,
        "trainable": overall["trainable"], "total": overall["total"],
        "trainable_pct": round(100 * overall["trainable"] / max(overall["total"], 1), 3),
        "backbone_trainable": bk["trainable"], "backbone_total": bk["total"],
        "backbone_trainable_pct": round(100 * bk["trainable"] / max(bk["total"], 1), 3),
        "train_step_peak_vram_mb": peak_mb,
        "loss_step0": round(float(loss.detach()), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/peft_lora.yaml")
    ap.add_argument("--methods", nargs="+",
                    default=["none", "lora", "adapter", "prompt", "full_ft"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print(f"PEFT SWEEP VERIFICATION  (config={args.config}, device={device})")
    print("  Teacher-feature cache in use? NO — teacher does a live ViT "
          "forward/backward; no precompute path exists in this codebase.")
    print("=" * 92)
    hdr = f"{'method':9} {'frames':>6} {'res':>4} {'bs':>3} {'trainable%':>11} {'bbone_train%':>13} {'peakVRAM_MB':>12}"
    print(hdr)
    print("-" * 92)
    rows = []
    for m in args.methods:
        r = measure_arm(m, cfg, device)
        rows.append(r)
        print(f"{r['method']:9} {r['frames']:>6} {r['resolution']:>4} "
              f"{r['batch_size']:>3} {r['trainable_pct']:>10.3f}% "
              f"{r['backbone_trainable_pct']:>12.3f}% {r['train_step_peak_vram_mb']:>12}")
    print("-" * 92)

    # Validity assertions.
    fs = {(r["frames"], r["resolution"], r["batch_size"]) for r in rows}
    print(f"Identical clip/res/batch across all arms? {'YES' if len(fs) == 1 else 'NO — MISMATCH!'} ({fs})")
    ff = next((r for r in rows if r["method"] == "full_ft"), None)
    if ff:
        genuine = ff["backbone_trainable_pct"] > 99.0
        print(f"full_ft backbone genuinely unfrozen (>99%)? "
              f"{'YES' if genuine else 'NO — backbone NOT fully trainable!'} "
              f"({ff['backbone_trainable_pct']}%)")
    return rows


if __name__ == "__main__":
    main()
