"""TerraMind surface-water model (needs `torch` and `terratorch`, see requirements-fm.txt).

  1. Labels come from the date-matched JRC monthly map (`labels.py`), not from a threshold on the radar itself.
  2. A pretrained TerraMind encoder (modality S1RTC = terrain-corrected gamma0 in dB, which is what Planetary Computer serves)
     reads VV and VH. The decoder also sees ALL input channels through a small convolutional stem: SDWI, relief, slope,
     aspect, the radar-look slope and the land-cover groups (`features.py`). That is a late fusion: the pretrained encoder
     is untouched, the terrain tells the decoder where a dark patch is shadow, and the stem keeps shorelines sharp.
  3. Land-cover channels are zeroed for a share of the training samples (`features.CONTEXT_DROPOUT`), so the model learns to
     use them as context only; water is decided by the radar.

TerraMind is a ViT encoder: input side must be a multiple of 16. Weights are fetched from Hugging Face
(`ibm-esa-geospatial/TerraMind-1.0-*`) when `pretrained=True`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as ft

MODALITY = "S1RTC"
PATCH = 16
# transformer layers whose tokens feed the decoder (4 evenly spaced taps)
TAPS = {"tiny": (2, 5, 8, 11), "small": (2, 5, 8, 11), "base": (2, 5, 8, 11), "large": (5, 11, 17, 23)}


def _size_of(backbone):
    for k in TAPS:
        if backbone.endswith(k):
            return k
    raise ValueError(f"unknown TerraMind size in {backbone!r}")


class TerraMindWater(nn.Module):
    """TerraMind S1RTC encoder + multi-layer token decoder + pixel stem -> 2-class logits at input resolution."""

    def __init__(self, backbone="terramind_v1_base", pretrained=True, freeze_backbone=True, unfreeze_last=0,
                 width=128, n_classes=2, ckpt_path=None, in_channels=ft.N_CHANNELS):
        super().__init__()
        self.backbone_name, self.width, self.n_classes, self.in_channels = backbone, width, n_classes, in_channels
        if in_channels < ft.N_ENCODER:
            raise ValueError(f"in_channels must be at least {ft.N_ENCODER} (VV, VH)")
        from terratorch.registry import BACKBONE_REGISTRY
        kw = {"ckpt_path": ckpt_path} if ckpt_path else {}
        self.enc = BACKBONE_REGISTRY.build(backbone, modalities=[MODALITY], pretrained=pretrained and not ckpt_path, **kw)
        self.taps = TAPS[_size_of(backbone)]
        dim = self.enc.out_channels[-1] if hasattr(self.enc, "out_channels") else self._probe_dim()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, width, 1), nn.BatchNorm2d(width), nn.GELU())
                                   for _ in self.taps])
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(width * len(self.taps) + 32, width, 3, padding=1), nn.BatchNorm2d(width),
                                  nn.GELU(), nn.Conv2d(width, n_classes, 1))
        self.set_trainable(freeze_backbone, unfreeze_last)

    def _probe_dim(self):
        with torch.no_grad():
            return self.enc({MODALITY: torch.zeros(1, ft.N_ENCODER, PATCH, PATCH)})[-1].shape[-1]

    def predict_prob(self, x, tile=224, overlap=64, batch=4, device=None):
        """Water probability (h, w) of a whole composed scene `x` (N_CHANNELS, h, w): see `predict_scene`."""
        return predict_scene(self, x, tile=tile, overlap=overlap, batch=batch, device=device)

    def set_trainable(self, freeze_backbone=True, unfreeze_last=0):
        """Freeze the encoder (default: only the decoder learns); `unfreeze_last` > 0 also trains the last blocks."""
        for p in self.enc.parameters():
            p.requires_grad = not freeze_backbone
        if freeze_backbone and unfreeze_last:
            blocks = getattr(self.enc, "encoder", None)
            blocks = list(blocks) if blocks is not None else []
            for blk in blocks[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def forward(self, x):
        """x: (B, in_channels, H, W) standardised (`features.compose`: VV, VH first); H, W multiples of 16."""
        b, ch, h, w = x.shape
        if ch != self.in_channels:
            raise ValueError(f"model expects {self.in_channels} channels, got {ch}")
        if h % PATCH or w % PATCH:
            raise ValueError(f"input {h}x{w} must be a multiple of {PATCH}")
        gh, gw = h // PATCH, w // PATCH
        frozen = not any(p.requires_grad for p in self.enc.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            feats = self.enc({MODALITY: x[:, :ft.N_ENCODER]})
        maps = []
        for proj, i in zip(self.proj, self.taps):
            t = feats[i].transpose(1, 2).reshape(b, -1, gh, gw)
            maps.append(F.interpolate(proj(t), size=(h, w), mode="bilinear", align_corners=False))
        return self.head(torch.cat(maps + [self.stem(x)], dim=1))


# ---------------------------------------------------------------------------------------- checkpoints
def _trained_state(model):
    """The weights that training changed: everything outside the encoder (decoder, pixel stem, head, BatchNorm statistics)
    plus any encoder parameters that were unfrozen. With the default frozen encoder this is ~4 MB for `base`, against
    ~350 MB for the whole model."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if not k.startswith("enc.") or k in trainable}


def save_checkpoint(model, path, meta=None, trainable_only=False):
    """Save a model plus what is needed to rebuild it (backbone name, decoder width) and free-form `meta`.

    trainable_only=False : the complete model, pretrained encoder included. Self-contained and loads offline, but large
                           (~350 MB for `terramind_v1_base`): keep it on Drive / a release, not in git (GitHub rejects > 100 MB).
    trainable_only=True  : only the weights training changed (see `_trained_state`), a few MB: fine for git. Loading it
                           rebuilds the pretrained encoder from Hugging Face (`load_checkpoint`).
    `meta` is stored as plain JSON-compatible data so the file loads with `torch.load(weights_only=True)`."""
    import json
    state = _trained_state(model) if trainable_only else model.state_dict()
    torch.save({"backbone": model.backbone_name, "width": model.width, "n_classes": model.n_classes,
                "in_channels": model.in_channels, "channels": list(ft.CHANNELS),
                "trainable_only": bool(trainable_only), "state_dict": state,
                "meta": json.loads(json.dumps(meta or {}, default=str))}, path)


def load_checkpoint(path, device="cpu", pretrained_backbone=True):
    """Rebuild a TerraMindWater from `save_checkpoint`. Returns (model in eval mode, meta).

    A full checkpoint loads without any download (the encoder weights are inside). A trainable-only checkpoint first builds the
    pretrained encoder from Hugging Face (`pretrained_backbone=True`, needs network) and then loads the saved weights on top;
    it refuses a file whose weights do not fit the model rather than silently leaving them random."""
    ck = torch.load(path, map_location=device, weights_only=True)
    light = ck.get("trainable_only", False)
    if ck.get("channels") != list(ft.CHANNELS):
        raise RuntimeError(f"{path}: trained with input channels {ck.get('channels', 'VV, VH only')} but this code feeds "
                           f"{list(ft.CHANNELS)}: retrain the model (notebooks/ai_water_training.ipynb).")
    model = TerraMindWater(ck["backbone"], pretrained=bool(light and pretrained_backbone), width=ck["width"],
                           n_classes=ck["n_classes"], in_channels=ck["in_channels"])
    res = model.load_state_dict(ck["state_dict"], strict=not light)
    if light:
        missing = [k for k in res.missing_keys if not k.startswith("enc.")]
        if missing or res.unexpected_keys:
            raise RuntimeError(f"{path}: checkpoint does not match {ck['backbone']} (width {ck['width']}): "
                               f"missing {missing[:3]}, unexpected {list(res.unexpected_keys)[:3]}")
    return model.to(device).eval(), ck.get("meta", {})


# ---------------------------------------------------------------------------------------- training
IGNORE = ft.IGNORE


class ArrayPatches:
    """Patches held as arrays X (n, C, s, s) float32, Y (n, s, s) uint8, with the same `batch(idx)` as `dataset.PatchSet`."""

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def batch(self, idx):
        idx = list(idx)
        return self.X[idx], self.Y[idx]


def dice_loss(logits, y, ignore=IGNORE, eps=1.0):
    valid = (y != ignore)
    p = logits.softmax(1)[:, 1] * valid
    t = (y == 1).float() * valid
    return 1.0 - (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def iou_water(logits, y, ignore=IGNORE):
    valid = y != ignore
    pred, tgt = (logits.argmax(1) == 1) & valid, (y == 1) & valid
    union = (pred | tgt).sum().item()
    return (pred & tgt).sum().item() / union if union else 1.0


def train(model, data, train_idx, val_idx, epochs=10, batch=8, lr=3e-4, device=None, log=print,
          context_dropout=ft.CONTEXT_DROPOUT, seed=0):
    """Fit the decoder (and any unfrozen blocks). `data` has `len()` and `batch(idx) -> (X, Y)` (`dataset.PatchSet`, or
    `ArrayPatches` for arrays). In `context_dropout` of the samples the land-cover channels are zeroed, so land cover stays a hint.
    There is no flip / rotation augmentation: aspect and the radar-look slope would no longer match the image.
    Returns the best-validation state (also loaded into `model`) and the history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(np.asarray(train_idx))
        tot = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            xb, yb = data.batch(idx)
            xb, yb = torch.from_numpy(np.ascontiguousarray(xb)).float(), torch.from_numpy(yb.astype(np.int64))
            if model.in_channels == ft.N_CHANNELS and context_dropout > 0:
                drop = torch.from_numpy(rng.random(len(idx)) < context_dropout)
                xb[drop, ft.LC_CH] = 0.0
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, ignore_index=IGNORE) + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        v = evaluate(model, data, val_idx, batch, device) if len(val_idx) else float("nan")
        hist.append({"epoch": ep, "loss": tot / max(1, len(order)), "val_iou": v})
        log(f"  epoch {ep + 1}/{epochs}  loss {hist[-1]['loss']:.4f}  val IoU(water) {v:.4f}")
        score = v if len(val_idx) else -hist[-1]["loss"]
        if score > best:
            best, best_state = score, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state, hist


@torch.no_grad()
def evaluate(model, data, idx, batch=8, device="cpu"):
    """IoU of the water class over `idx` (labelled pixels only), all channels given."""
    model.eval()
    inter = union = 0
    for s in range(0, len(idx), batch):
        xb, yb = data.batch(list(idx[s:s + batch]))
        logits = model(torch.from_numpy(np.ascontiguousarray(xb)).float().to(device))
        y = torch.from_numpy(yb.astype(np.int64)).to(device)
        valid = y != IGNORE
        pred, tgt = (logits.argmax(1) == 1) & valid, (y == 1) & valid
        inter += (pred & tgt).sum().item()
        union += (pred | tgt).sum().item()
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_scene(model, x, tile=224, overlap=64, batch=4, device=None):
    """Water probability (h, w) float32 for a whole scene by overlapping tiles (Hann-weighted blending).
    `x`: (N_CHANNELS, h, w) from `features.compose`. tile must be a multiple of 16; image edges are reflect-padded."""
    device = device or next(model.parameters()).device
    model.eval().to(device)
    x = np.asarray(x, np.float32)
    _, h, w = x.shape
    step = tile - overlap
    ph = (-(h - tile) % step) if h > tile else tile - h
    pw = (-(w - tile) % step) if w > tile else tile - w
    xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    H, W = xp.shape[1:]
    win = np.outer(np.hanning(tile + 2)[1:-1], np.hanning(tile + 2)[1:-1]).astype(np.float32) + 1e-3
    acc, wsum = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    corners = [(r, c) for r in range(0, H - tile + 1, step) for c in range(0, W - tile + 1, step)]
    for s in range(0, len(corners), batch):
        cs = corners[s:s + batch]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in cs])).to(device)
        pw_ = model(xb).softmax(1)[:, 1].cpu().numpy()
        for (r, c), p in zip(cs, pw_):
            acc[r:r + tile, c:c + tile] += p * win
            wsum[r:r + tile, c:c + tile] += win
    return (acc / np.maximum(wsum, 1e-6))[:h, :w]
