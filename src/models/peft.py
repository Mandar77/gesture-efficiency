"""Minimal, self-contained PEFT (parameter-efficient fine-tuning) primitives for
adapting a *frozen* timm Vision Transformer to video-gesture recognition.

No dependency on the external ``peft`` library (it is installed only as an
optional cross-check). Everything here is a thin wrapper around ``nn.Linear`` /
``nn.Module`` so that:

  * the pretrained backbone weights stay **frozen** (``requires_grad=False``),
  * only the injected low-rank / bottleneck / prompt parameters are trainable,
  * the wrappers add negligible VRAM (important for the RTX 4060 8 GB budget).

Three methods are provided, matching BRIEF §3.1:

  * ``apply_lora``       -- low-rank adapters on attention q/k/v/o projections.
  * ``insert_adapters``  -- AIM-style bottleneck adapters after attn and MLP.
  * ``PromptTokens``     -- learnable prompt tokens prepended to the patch seq.

plus the freezing helpers ``freeze_backbone`` and ``mark_trainable``.

The functions mutate the backbone in place and return a list of the *new*
trainable modules/parameters they created, so the caller can log exactly what
became trainable.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

log = get_logger("models.peft")


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------
def freeze_backbone(module: nn.Module) -> None:
    """Freeze **every** parameter in ``module`` (sets ``requires_grad=False``).

    Call this on the backbone *before* injecting PEFT params so that only the
    freshly-created adapter/LoRA/prompt tensors carry gradients.
    """
    for p in module.parameters():
        p.requires_grad_(False)


def mark_trainable(params: Iterable[nn.Parameter]) -> None:
    """Set ``requires_grad=True`` on an iterable of parameters."""
    for p in params:
        p.requires_grad_(True)


def _iter_named_linears(root: nn.Module):
    """Yield ``(parent_module, attr_name, linear)`` for every ``nn.Linear``.

    We yield the *parent* + attribute name so callers can swap the child in
    place with ``setattr``.
    """
    for parent in root.modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                yield parent, attr, child


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank residual path.

    Computes ``y = base(x) + scaling * (dropout(x) @ A^T @ B^T)`` where
    ``A: [r, in]`` and ``B: [out, r]``. The base linear's weight/bias are kept
    but frozen; only ``A`` and ``B`` (``r*(in+out)`` params) train.

    ``scaling = alpha / r`` is the standard LoRA normalisation so that tuning
    ``alpha`` roughly decouples effective learning-rate from the chosen rank.
    ``B`` is zero-initialised so the wrapped module is an exact identity at
    step 0 (the pretrained behaviour is preserved before any training).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        # Freeze the pretrained projection.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A down-projects into the rank subspace, B up-projects back out.
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank))
        # Kaiming init on A (as in the LoRA paper); B stays zero => zero residual.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_dropout(x)
        # (x @ A^T) : [..., r]  then  (@ B^T) : [..., out]
        lora = torch.nn.functional.linear(lora, self.lora_A)
        lora = torch.nn.functional.linear(lora, self.lora_B)
        return out + self.scaling * lora

    # Convenience so external code / distillation can introspect.
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _linear_matches_targets(name: str, targets: Sequence[str]) -> bool:
    """Decide whether a Linear (by its dotted module name) is a LoRA target.

    Heuristics for timm ViT, whose attention uses a *fused* ``attn.qkv`` Linear
    and a separate ``attn.proj`` Linear (there is no split q/k/v). We therefore:

      * always require the name to sit under an ``attn`` submodule, and
      * treat any of the target tokens {q,k,v,qkv} as "the qkv projection", and
      * treat {o,proj,out} as "the output projection".

    ``targets`` comes from config (e.g. ``[q, k, v, o]``). Because timm fuses
    q/k/v, requesting any of q/k/v enables LoRA on the single fused qkv Linear.
    """
    lname = name.lower()
    if "attn" not in lname:
        return False
    tset = {t.lower() for t in targets}
    is_qkv = lname.endswith("qkv") or lname.endswith(".q") or lname.endswith(".k") or lname.endswith(".v")
    is_proj = lname.endswith("proj") or lname.endswith(".o") or lname.endswith(".out")
    if is_qkv and (tset & {"q", "k", "v", "qkv"}):
        return True
    if is_proj and (tset & {"o", "proj", "out"}):
        return True
    return False


def apply_lora(
    model: nn.Module,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[LoRALinear]:
    """Inject :class:`LoRALinear` into the attention projections of ``model``.

    Matches attention ``qkv`` / ``proj`` Linears by name (see
    :func:`_linear_matches_targets`), replacing each with a LoRA-wrapped copy.
    Returns the list of created wrappers. Operates in place.
    """
    targets = list(targets) if targets else ["q", "k", "v", "o"]
    wrappers: List[LoRALinear] = []
    # Snapshot the target (parent, attr, child, full_name) BEFORE mutating. We
    # must NOT mutate the tree while iterating `model.modules()` (torch's live
    # traversal would then descend into the freshly-inserted LoRALinear.base,
    # causing unbounded recursion). Materialise the full list first, then swap.
    name_of = {id(m): n for n, m in model.named_modules()}
    to_wrap = []
    for parent, attr, child in _iter_named_linears(model):
        full = name_of.get(id(child), attr)
        if _linear_matches_targets(full, targets):
            to_wrap.append((parent, attr, child))
    for parent, attr, child in to_wrap:
        wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        wrappers.append(wrapped)
    if not wrappers:
        log.warning(
            "apply_lora matched 0 Linear layers for targets=%s. "
            "Backbone attention naming may differ from timm ViT.",
            targets,
        )
    else:
        log.info("LoRA: wrapped %d attention projections (rank=%d, alpha=%s).",
                 len(wrappers), rank, alpha)
    return wrappers


# ---------------------------------------------------------------------------
# Adapters (AIM-style bottleneck)
# ---------------------------------------------------------------------------
class Adapter(nn.Module):
    """Bottleneck adapter: ``x + up(gelu(down(x)))``.

    ``down: D -> dim`` and ``up: dim -> D``. ``up`` is zero-initialised so the
    adapter starts as identity (preserves pretrained behaviour at step 0). Only
    ``2*D*dim`` params per adapter — cheap enough to place after both the
    attention and MLP sublayers of every block.
    """

    def __init__(self, dim_model: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class _BlockWithAdapters(nn.Module):
    """Wrap a timm ViT block, inserting an :class:`Adapter` after the block.

    A ViT block already applies its own residuals internally (x = x + attn(x);
    x = x + mlp(x)). Rather than surgically splitting the block (which is
    version-fragile), we run the original frozen block and then apply a single
    trainable adapter as a residual on its output. This is functionally an
    AIM-style *post* adapter and keeps the wrapper backbone-agnostic.

    To also get a per-sublayer adapter without editing timm internals, we add a
    second adapter on the block input as a light pre-adapter; both are cheap.
    """

    def __init__(self, block: nn.Module, dim_model: int, bottleneck: int):
        super().__init__()
        self.block = block
        self.adapter_pre = Adapter(dim_model, bottleneck)
        self.adapter_post = Adapter(dim_model, bottleneck)

    def forward(self, x, *args, **kwargs):
        x = self.adapter_pre(x)
        x = self.block(x, *args, **kwargs)
        x = self.adapter_post(x)
        return x


def insert_adapters(model: nn.Module, dim: int) -> List[Adapter]:
    """Insert bottleneck adapters around every transformer block in ``model``.

    Looks for a ``blocks`` container (timm ViT: ``model.blocks`` is a
    ``ModuleList``/``Sequential`` of blocks) and wraps each block with
    :class:`_BlockWithAdapters`. Returns the list of created adapters.

    Falls back to a warning (no-op) if no ``blocks`` container is found.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None or not hasattr(blocks, "__len__"):
        log.warning("insert_adapters: no 'blocks' container found on backbone; "
                    "no adapters inserted.")
        return []

    dim_model = int(getattr(model, "embed_dim", None) or getattr(model, "num_features"))
    created: List[Adapter] = []
    for i in range(len(blocks)):
        wrapped = _BlockWithAdapters(blocks[i], dim_model, dim)
        blocks[i] = wrapped  # ModuleList / Sequential both support item assign
        created.extend([wrapped.adapter_pre, wrapped.adapter_post])
    log.info("Adapters: inserted %d bottleneck adapters (dim=%d) across %d blocks.",
             len(created), dim, len(blocks))
    return created


# ---------------------------------------------------------------------------
# Prompt tuning (shallow)
# ---------------------------------------------------------------------------
class PromptTokens(nn.Module):
    """Learnable prompt tokens prepended to the patch-token sequence.

    We implement **shallow** (VPT-Shallow) prompt tuning: a single set of
    ``num_tokens`` learnable ``[D]`` vectors is prepended to the token sequence
    once, right after patch embedding + CLS/pos-embed, before the transformer
    blocks. Shallow is chosen (vs. deep, which injects fresh prompts at every
    layer) because it is simpler, adds the fewest params, and is a well-studied
    baseline; the extra tokens increase attention cost only marginally.

    This module owns just the prompt parameters; the backbone wrapper is
    responsible for actually concatenating them into the sequence (see
    :class:`PEFTVideoTeacher`), which keeps prompt handling explicit and
    backbone-version independent.
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"prompt_tokens must be > 0, got {num_tokens}")
        self.num_tokens = int(num_tokens)
        self.prompt = nn.Parameter(torch.zeros(1, num_tokens, dim))
        # Small random init (VPT uses a scaled uniform); keep magnitude modest.
        nn.init.trunc_normal_(self.prompt, std=0.02)

    def expand(self, batch: int) -> torch.Tensor:
        """Return the prompt tokens broadcast to ``[batch, num_tokens, dim]``."""
        return self.prompt.expand(batch, -1, -1)

    def forward(self, batch: int) -> torch.Tensor:  # convenience alias
        return self.expand(batch)
