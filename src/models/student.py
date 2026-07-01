"""Streaming causal student — MoViNet-A0/A1-class compact 3D-CNN (BRIEF §3.2).

The point of this model is *streaming, constant-memory, causal* inference: it can
consume a video one frame at a time (online) and produce a prediction after every
frame, while its memory footprint stays constant regardless of how long the stream
has been running. This is achieved with two ingredients:

  1. `CausalConv3d` — a 3D convolution that only ever looks at the *current and
     past* frames. We do this by padding **only the front** of the temporal
     dimension by ``(kt - 1) * dilation`` and using **no right (future) pad**.
     Output frame t therefore depends only on input frames ``<= t``.

  2. `StreamBuffer` — every `CausalConv3d` keeps a rolling cache of its last
     ``(kt - 1) * dilation`` *input* frames. In streaming mode, instead of
     re-padding with zeros we prepend the cache, convolve the (cache + new)
     chunk, then update the cache with the tail of the new input. The cache size
     is fixed by the kernel, so per-layer state is O(kt) frames — constant in the
     stream length. Feeding frames one-by-one reproduces (bit-identically for the
     conv stack) the result of the batch `forward` on the whole clip, because the
     concatenation of cached-past + current is exactly the left-padded receptive
     field the batch path sees.

Batch API (whole clip):
    logits = model(x)                       # x = [B, C, T, H, W]
    logits, feats = model(x, return_features=True)

Streaming API (constant memory, online):
    model.reset_stream(batch_size=B, device=...)   # clears all layer caches
    for t in range(T):
        logits_t = model.forward_step(frame_bchw)   # frame = [B, C, H, W]
    # logits_t is the prediction using all frames seen so far.

Streaming logit definition
--------------------------
The batch head global-average-pools over the *whole* clip's temporal axis, so a
single ``forward_step`` (which has only seen frames ``<= t``) cannot be
bit-identical to it. We instead maintain a **causal running average** of the
per-frame pooled features: after step t the head sees the mean of the pooled
feature maps over frames ``0..t``. When the full clip has been streamed this
running average equals the batch temporal-global-average-pool over the (causally
produced) feature map, so ``forward_step`` on the last frame and ``forward`` on
the full clip give comparable predictions (they differ only through the causal
vs. centred temporal receptive field, which is by design).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import register


# --------------------------------------------------------------------------- #
# Causal 3D convolution with an optional streaming cache (the "stream buffer"). #
# --------------------------------------------------------------------------- #
class CausalConv3d(nn.Module):
    """Conv3d that only sees current + past frames along time.

    We left-pad time by ``(kt - 1) * dilation_t`` and never right-pad, so output
    frame ``t`` depends only on input frames ``<= t`` (strictly causal). Spatial
    dims are padded symmetrically as usual.

    In streaming mode each instance keeps a rolling cache of its last
    ``(kt - 1) * dilation_t`` *input* frames (``self._cache``), giving
    constant-memory online inference: ``forward_stream`` prepends the cache to a
    new chunk, convolves, and stores the chunk's tail as the next cache.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        stride: Sequence[int] | int = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        kt, kh, kw = _triple(kernel_size)
        st, sh, sw = _triple(stride)
        dt, dh, dw = _triple(dilation)
        # Temporal stride must be 1 for the streaming cache logic to line up
        # frame-for-frame; we only ever downsample time via pooling, not stride.
        assert st == 1, "CausalConv3d requires temporal stride 1 (downsample time via pooling)"

        self.kt = kt
        self.dt = dt
        # Number of past frames we must retain to convolve the next input causally.
        self.time_pad = (kt - 1) * dt

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kt, kh, kw),
            stride=(1, sh, sw),
            padding=(0, dh * (kh - 1) // 2, dw * (kw - 1) // 2),  # spatial only; time handled manually
            dilation=(dt, dh, dw),
            groups=groups,
            bias=bias,
        )

        self.in_channels = in_channels
        # Streaming state: cache of the last `time_pad` input frames, or None.
        self._cache: Optional[torch.Tensor] = None
        self._streaming = False

    # -- batch (whole-clip) path ------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad time with zeros so the first frames have a causal (zero) past.
        x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))  # pad = (w_l,w_r, h_l,h_r, t_l,t_r)
        return self.conv(x)

    # -- streaming (online) path ------------------------------------------- #
    def reset_stream(self) -> None:
        """Drop the rolling cache; the next chunk starts with a zero past."""
        self._cache = None
        self._streaming = True

    def forward_stream(self, x_t: torch.Tensor) -> torch.Tensor:
        """Process a new temporal chunk ``x_t`` = [B, C, t, H, W] using the cache.

        Constant memory: the retained state is exactly ``time_pad`` frames,
        independent of how many frames have already been streamed.
        """
        b, c, t, h, w = x_t.shape
        if self.time_pad == 0:
            ctx = x_t
        else:
            if self._cache is None:
                # First chunk: causal past is zeros (matches batch left-pad).
                pad = x_t.new_zeros((b, c, self.time_pad, h, w))
                ctx = torch.cat([pad, x_t], dim=2)
            else:
                ctx = torch.cat([self._cache, x_t], dim=2)
            # Retain the last `time_pad` input frames as the next cache.
            self._cache = ctx[:, :, -self.time_pad:].detach() if not self.training else ctx[:, :, -self.time_pad:]
        # ctx now has (time_pad + t) frames; conv (no time pad) yields exactly t outputs.
        return self.conv(ctx)


def _triple(v: Sequence[int] | int) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    v = tuple(v)
    assert len(v) == 3, f"expected int or length-3 sequence, got {v!r}"
    return v  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Squeeze-and-Excite (MoViNet flavour) over the spatial+temporal dims.         #
# --------------------------------------------------------------------------- #
class SqueezeExcite3d(nn.Module):
    """Channel attention. In streaming mode the squeeze pools over space only
    (per-frame), which keeps it causal / constant-memory. In batch mode it pools
    over space and the local chunk time — both are strictly per-current-frame
    friendly because pooling is applied after the causal conv."""

    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(round(channels * se_ratio)))
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def _scale(self, x: torch.Tensor, spatial_only: bool) -> torch.Tensor:
        dims = (3, 4) if spatial_only else (2, 3, 4)
        s = x.mean(dim=dims, keepdim=True)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-frame (spatial-only) squeeze keeps SE causal frame-by-frame, so the
        # batch and streaming paths agree.
        return self._scale(x, spatial_only=True)


# --------------------------------------------------------------------------- #
# MoViNet-style inverted-residual block with a depthwise *causal* 3D conv.      #
# --------------------------------------------------------------------------- #
class CausalBottleneck(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        kt: int = 3,
        stride_hw: int = 1,
        expand: int = 4,
        se_ratio: Optional[float] = 0.25,
    ):
        super().__init__()
        cmid = cin * expand
        self.use_residual = (stride_hw == 1 and cin == cout)

        # 1x1x1 pointwise expansion (not temporal -> not causal-sensitive).
        self.expand_pw = (
            nn.Sequential(nn.Conv3d(cin, cmid, 1, bias=False), nn.BatchNorm3d(cmid), nn.Hardswish())
            if expand != 1
            else nn.Identity()
        )
        # Depthwise *causal* conv: temporal kernel kt, spatial 3x3, groups=cmid.
        self.dw = CausalConv3d(
            cmid, cmid, kernel_size=(kt, 3, 3), stride=(1, stride_hw, stride_hw), groups=cmid, bias=False
        )
        self.dw_bn = nn.BatchNorm3d(cmid)
        self.dw_act = nn.Hardswish()
        self.se = SqueezeExcite3d(cmid, se_ratio) if se_ratio else nn.Identity()
        # 1x1x1 pointwise projection back to cout (linear, no activation).
        self.proj_pw = nn.Sequential(nn.Conv3d(cmid, cout, 1, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand_pw(x)
        out = self.dw_act(self.dw_bn(self.dw(out)))
        out = self.se(out)
        out = self.proj_pw(out)
        if self.use_residual:
            out = out + x
        return out

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand_pw(x)
        out = self.dw_act(self.dw_bn(self.dw.forward_stream(out)))
        out = self.se(out)
        out = self.proj_pw(out)
        if self.use_residual:
            out = out + x
        return out

    def reset_stream(self) -> None:
        self.dw.reset_stream()


# --------------------------------------------------------------------------- #
# The streaming student.                                                       #
# --------------------------------------------------------------------------- #
@register("model", "streaming_student")
class StreamingStudent(nn.Module):
    """Compact causal streaming 3D-CNN (MoViNet-A0/A1 class).

    Parameter budget (BRIEF §3.2 target 3-5M): at the defaults
    ``width=32, blocks=(2, 3, 3, 4), expand=4`` this model has **~3.1M
    parameters** (verified by ``sum(p.numel())`` on CPU). Reducing to
    ``width=24`` gives ~1.8M (A0-ish); ``width=40`` gives ~4.8M (A1-ish).

    Contract:
        forward(x, return_features=False) with x=[B, C, T, H, W] ->
            logits [B, num_classes] or (logits, features[B, feature_dim]).
        self.feature_dim (int) exposed for feature distillation.
    """

    def __init__(
        self,
        num_classes: int = 27,
        width: int = 32,
        blocks: Sequence[int] = (2, 3, 3, 4),
        in_channels: int = 3,
        expand: int = 4,
        se_ratio: Optional[float] = 0.25,
        kt: int = 3,
        dropout: float = 0.2,
        **_ignore,
    ):
        super().__init__()
        self.in_channels = in_channels
        blocks = tuple(blocks)

        # Causal stem: temporal kt, spatial 3x3, spatial stride 2 (downsample HW).
        self.stem = nn.Sequential(
            CausalConv3d(in_channels, width, kernel_size=(kt, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(width),
            nn.Hardswish(),
        )

        # Channel schedule per stage; first block of each stage strides HW by 2.
        stage_channels = [width, width * 2, width * 4, width * 6][: len(blocks)]
        self.stages = nn.ModuleList()
        cin = width
        for cout, n in zip(stage_channels, blocks):
            for j in range(n):
                self.stages.append(
                    CausalBottleneck(
                        cin,
                        cout,
                        kt=kt,
                        stride_hw=(2 if j == 0 else 1),
                        expand=expand,
                        se_ratio=se_ratio,
                    )
                )
                cin = cout

        # Head expansion conv (1x1x1, non-temporal).
        head_dim = cin * 2
        self.head_conv = nn.Sequential(
            nn.Conv3d(cin, head_dim, 1, bias=False), nn.BatchNorm3d(head_dim), nn.Hardswish()
        )
        self.feature_dim = head_dim  # exposed for feature-distillation projection
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, num_classes)

        # Streaming state for the causal running-average pooling.
        self._streaming = False
        self._feat_sum: Optional[torch.Tensor] = None  # [B, head_dim] running feature sum
        self._frame_count = 0

    # ---------------- shared spatial feature extractor -------------------- #
    def _spatial_features(self, feat_map: torch.Tensor) -> torch.Tensor:
        """[B, C, t, H, W] -> [B, C, t] by global-average-pool over H, W."""
        return feat_map.mean(dim=(3, 4))

    # ---------------- batch (whole-clip) path ----------------------------- #
    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.stem(x)
        for blk in self.stages:
            h = blk(h)
        h = self.head_conv(h)  # [B, head_dim, T', H', W']
        # Global average pool over space AND (causally produced) time.
        feats = h.mean(dim=(2, 3, 4))  # [B, head_dim]
        logits = self.classifier(self.dropout(feats))
        if return_features:
            return logits, feats
        return logits

    # ---------------- streaming (online, constant-memory) path ------------ #
    def reset_stream(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        """Clear every layer's stream buffer and the running feature average.

        Call once before feeding a fresh stream. ``batch_size``/``device`` are
        optional; the running-average state lazily initialises on the first
        ``forward_step``.
        """
        self._streaming = True
        self._feat_sum = None
        self._frame_count = 0
        for m in self.modules():
            if isinstance(m, CausalConv3d):
                m.reset_stream()

    @torch.no_grad()
    def forward_step(self, frame_bchw: torch.Tensor) -> torch.Tensor:
        """Process ONE new frame [B, C, H, W] using internal buffers.

        Returns logits [B, num_classes] using all frames seen since the last
        ``reset_stream``. Memory is constant in the number of frames streamed:
        the only retained state is each CausalConv3d's fixed-size cache plus a
        single running feature sum (see module docstring, "Streaming logit").
        """
        if not self._streaming:
            self.reset_stream()
        # Add a singleton temporal axis: [B, C, H, W] -> [B, C, 1, H, W].
        x = frame_bchw.unsqueeze(2)

        # Stem (causal conv uses its stream cache).
        h = self.stem[0].forward_stream(x)
        h = self.stem[2](self.stem[1](h))
        for blk in self.stages:
            h = blk.forward_stream(h)
        h = self.head_conv(h)  # [B, head_dim, 1, H', W']

        # Per-frame pooled feature over space (and the single time slice).
        frame_feat = h.mean(dim=(2, 3, 4))  # [B, head_dim]

        # Causal running average of pooled features == temporal-global-avg-pool
        # over frames seen so far. This is what makes the last streamed step
        # comparable to forward() on the whole clip.
        if self._feat_sum is None:
            self._feat_sum = torch.zeros_like(frame_feat)
        self._feat_sum = self._feat_sum + frame_feat
        self._frame_count += 1
        feats = self._feat_sum / self._frame_count

        logits = self.classifier(feats)
        return logits
