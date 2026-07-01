"""Real-time webcam gesture demo using the streaming student (BRIEF §7, M8).

Drives the streaming student's constant-memory `forward_step` API: each captured
frame is preprocessed and fed one at a time, so memory is constant regardless of
how long the demo runs — this is the on-device real-time claim made visible.

On-screen overlay reports measured FPS and end-to-end per-frame latency (the
numbers the paper's frontier reports), reusing the proven rolling-average FPS
pattern from the base project's live_recognition harness.

Usage:
    python src/demo/webcam_demo.py --ckpt checkpoints/distill/student.pt \
        --labels data/jester/jester-v1-labels.csv --frame-size 172

If no checkpoint is given, runs with a randomly-initialised student so the
pipeline/overlay can be demoed without trained weights (predictions meaningless;
clearly labelled DEMO-UNTRAINED on screen).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.utils.logging_utils import get_logger

log = get_logger("demo.webcam")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_labels(path: str | None):
    if path and Path(path).exists():
        return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return None


def build_student(ckpt: str | None, num_classes: int, device, frame_size: int):
    import src.models  # noqa: F401  populate registry
    from src.utils.registry import build

    model = None
    if ckpt and Path(ckpt).exists():
        from src.utils.checkpoint import load_checkpoint

        payload = load_checkpoint(ckpt, map_location="cpu")
        cfg = payload.get("config") or {}
        mkwargs = dict((cfg.get("model", {}) or {}).get("kwargs", {}))
        mkwargs.setdefault("num_classes", (cfg.get("data", {}) or {}).get("num_classes", num_classes))
        name = (cfg.get("model", {}) or {}).get("name", "streaming_student")
        model = build("model", name, **mkwargs)
        model.load_state_dict(payload["model_state"], strict=False)
        num_classes = mkwargs["num_classes"]
        log.info("Loaded student '%s' from %s", name, ckpt)
    else:
        model = build("model", "streaming_student", num_classes=num_classes)
        log.warning("No checkpoint — running DEMO-UNTRAINED (predictions meaningless).")
    return model.to(device).eval(), num_classes


def preprocess(frame_bgr, size: int, device) -> torch.Tensor:
    import cv2

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)  # [1,3,H,W]
    return t.to(device)


@torch.no_grad()
def main():
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--frame-size", type=int, default=172)
    ap.add_argument("--num-classes", type=int, default=27)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--show-hand", action="store_true",
                    help="Overlay MediaPipe hand landmarks (visual only).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(args.labels)
    model, num_classes = build_student(args.ckpt, args.num_classes, device, args.frame_size)
    trained = args.ckpt and Path(args.ckpt).exists()

    # Streaming: reset the causal buffers once, then feed frames one-by-one.
    if hasattr(model, "reset_stream"):
        model.reset_stream()
        stream = True
    else:
        stream = False
        log.warning("Model has no forward_step; falling back to clip buffering.")
    clip_buf = deque(maxlen=16)

    hands = None
    if args.show_hand:
        try:
            from mediapipe import solutions

            hands = solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.6)
            mp_draw = solutions.drawing_utils
        except Exception as e:  # pragma: no cover
            log.warning("MediaPipe unavailable (%s); disabling hand overlay.", e)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera %d.", args.camera)
        return
    frame_times = deque(maxlen=30)
    infer_ms = deque(maxlen=30)
    log.info("Running on %s. Press 'q' to quit.", device)

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        x = preprocess(frame, args.frame_size, device)
        ti = time.perf_counter()
        if stream:
            logits = model.forward_step(x)
        else:
            clip_buf.append(x.squeeze(0))
            clip = torch.stack(list(clip_buf), dim=1).unsqueeze(0)  # [1,C,T,H,W]
            logits = model(clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_ms.append((time.perf_counter() - ti) * 1000.0)

        probs = torch.softmax(logits.float(), dim=1)[0]
        conf, idx = probs.max(0)
        label = labels[idx] if labels and idx < len(labels) else f"class {int(idx)}"

        if hands is not None:
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, solutions.hands.HAND_CONNECTIONS)

        frame_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))
        lat = sum(infer_ms) / len(infer_ms)

        banner = f"{label} ({conf:.2f})" if trained else "DEMO-UNTRAINED"
        cv2.putText(frame, banner, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"infer: {lat:.1f} ms", (frame.shape[1] - 190, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{device.type}", (frame.shape[1] - 90, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("gesture-efficiency demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
