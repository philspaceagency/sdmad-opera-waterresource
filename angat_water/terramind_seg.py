"""TerraMind refinement of the Otsu water maps (needs `torch` and `terratorch`, see requirements-fm.txt).

Pipeline
  1. Otsu (edge + binary) gives pseudo-labels (`pseudo_labels.py`): certain water / certain land; the uncertain
     band is ignored in the loss.
  2. A pretrained TerraMind encoder (modality S1RTC = terrain-corrected gamma0 in dB, which is what Planetary
     Computer serves) is fine-tuned with a light decoder. The ViT works on 16 px tokens, so the decoder also sees
     the pixels themselves (a small convolutional stem on the 2 input channels) to keep shorelines sharp.
  3. `fuse` combines the model with Otsu: the thresholds supply the precise boundary where they are confident, the
     model settles the pixels the thresholds are unsure about, and the disagreements are reported, not hidden.

TerraMind is a ViT encoder: input side must be a multiple of 16. Weights are fetched from Hugging Face
(`ibm-esa-geospatial/TerraMind-1.0-*`) when `pretrained=True`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pseudo_labels as pl

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
                 width=128, n_classes=2, ckpt_path=None):
        super().__init__()
        self.backbone_name, self.width, self.n_classes = backbone, width, n_classes
        from terratorch.registry import BACKBONE_REGISTRY
        kw = {"ckpt_path": ckpt_path} if ckpt_path else {}
        self.enc = BACKBONE_REGISTRY.build(backbone, modalities=[MODALITY], pretrained=pretrained and not ckpt_path, **kw)
        self.taps = TAPS[_size_of(backbone)]
        dim = self.enc.out_channels[-1] if hasattr(self.enc, "out_channels") else self._probe_dim()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, width, 1), nn.BatchNorm2d(width), nn.GELU())
                                   for _ in self.taps])
        self.stem = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(width * len(self.taps) + 32, width, 3, padding=1), nn.BatchNorm2d(width),
                                  nn.GELU(), nn.Conv2d(width, n_classes, 1))
        self.set_trainable(freeze_backbone, unfreeze_last)

    def _probe_dim(self):
        with torch.no_grad():
            return self.enc({MODALITY: torch.zeros(1, 2, PATCH, PATCH)})[-1].shape[-1]

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
        """x: (B, 2, H, W) standardised dB (VV, VH); H, W multiples of 16."""
        b, _, h, w = x.shape
        if h % PATCH or w % PATCH:
            raise ValueError(f"input {h}x{w} must be a multiple of {PATCH}")
        gh, gw = h // PATCH, w // PATCH
        frozen = not any(p.requires_grad for p in self.enc.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            feats = self.enc({MODALITY: x})
        maps = []
        for proj, i in zip(self.proj, self.taps):
            t = feats[i].transpose(1, 2).reshape(b, -1, gh, gw)
            maps.append(F.interpolate(proj(t), size=(h, w), mode="bilinear", align_corners=False))
        return self.head(torch.cat(maps + [self.stem(x)], dim=1))


# ---------------------------------------------------------------------------------------- checkpoints
def save_checkpoint(model, path, meta=None):
    """Weights + what is needed to rebuild the model (backbone name, decoder width) + free-form `meta`."""
    torch.save({"backbone": model.backbone_name, "width": model.width, "n_classes": model.n_classes,
                "state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a TerraMindWater from `save_checkpoint` WITHOUT downloading the pretrained backbone weights (they are
    inside the checkpoint). Returns (model in eval mode, meta)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TerraMindWater(ck["backbone"], pretrained=False, width=ck["width"], n_classes=ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("meta", {})


# ---------------------------------------------------------------------------------------- training
def dice_loss(logits, y, ignore=pl.IGNORE, eps=1.0):
    valid = (y != ignore)
    p = logits.softmax(1)[:, 1] * valid
    t = (y == 1).float() * valid
    return 1.0 - (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def iou_water(logits, y, ignore=pl.IGNORE):
    valid = y != ignore
    pred, tgt = (logits.argmax(1) == 1) & valid, (y == 1) & valid
    union = (pred | tgt).sum().item()
    return (pred & tgt).sum().item() / union if union else 1.0


def train(model, X, Y, train_idx, val_idx, epochs=10, batch=8, lr=3e-4, device=None, log=print, flip=True):
    """Fit the decoder (and any unfrozen blocks) on pseudo-label patches. X (n,2,s,s) float32, Y (n,s,s) uint8.
    Returns the best-validation state (also loaded into `model`) and the history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y.astype(np.int64))
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(train_idx)
        tot = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            if flip and np.random.rand() < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            if flip and np.random.rand() < 0.5:
                xb, yb = xb.flip(-2), yb.flip(-2)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, ignore_index=pl.IGNORE) + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        v = evaluate(model, Xt, Yt, val_idx, batch, device) if len(val_idx) else float("nan")
        hist.append({"epoch": ep, "loss": tot / max(1, len(order)), "val_iou": v})
        log(f"  epoch {ep + 1}/{epochs}  loss {hist[-1]['loss']:.4f}  val IoU(water) {v:.4f}")
        score = v if len(val_idx) else -hist[-1]["loss"]
        if score > best:
            best, best_state = score, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state, hist


@torch.no_grad()
def evaluate(model, Xt, Yt, idx, batch=8, device="cpu"):
    model.eval()
    inter = union = 0
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        logits = model(Xt[i].to(device))
        y = Yt[i].to(device)
        valid = y != pl.IGNORE
        pred, tgt = (logits.argmax(1) == 1) & valid, (y == 1) & valid
        inter += (pred & tgt).sum().item()
        union += (pred | tgt).sum().item()
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_scene(model, vv_db, vh_db, tile=224, overlap=64, batch=4, device=None):
    """Water probability (h, w) float32 for a whole scene by overlapping tiles (Hann-weighted blending).
    tile must be a multiple of 16; image edges are reflect-padded."""
    device = device or next(model.parameters()).device
    model.eval().to(device)
    x = pl.normalise_s1(vv_db, vh_db)
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


# ---------------------------------------------------------------------------------------- fusion with Otsu
FUSE_LAND, FUSE_WATER, FUSE_FM_ONLY, FUSE_OTSU_ONLY = 0, 1, 2, 3


def fuse(prob, classes, sure_conflict=0.9, accept=0.5):
    """Combine the TerraMind water probability with the Otsu classes.

      * Otsu certain water, model not strongly against it  -> water
      * Otsu uncertain band                                 -> model decides (prob >= accept)
      * Otsu certain land, model sure of water (>= sure_conflict) -> NOT water, but flagged FUSE_FM_ONLY
        (candidate flooded vegetation / missed shallow water: review, do not trust blindly)
      * Otsu certain water, model sure it is land (<= 1 - sure_conflict) -> water, flagged FUSE_OTSU_ONLY
    Returns (water bool, flags uint8 with 0 land, 1 water, 2 fm_only, 3 otsu_only).
    """
    from . import otsu
    sure_w, sure_l = classes == otsu.CLASS_WATER, classes == otsu.CLASS_LAND
    unc = classes == otsu.CLASS_UNCERTAIN
    water = sure_w | (unc & (prob >= accept))
    flags = np.zeros(classes.shape, np.uint8)
    flags[water] = FUSE_WATER
    flags[sure_l & (prob >= sure_conflict)] = FUSE_FM_ONLY
    flags[sure_w & (prob <= 1.0 - sure_conflict)] = FUSE_OTSU_ONLY
    return water, flags


def refine_threshold(db, prob, hi=0.9, lo=0.1, min_px=500):
    """FM-guided Otsu: re-estimate the dB threshold from pixels the model is sure about. Water / land are modelled as
    Gaussians on confident pixels and the threshold is where both are the same number of standard deviations away
    (t = (mu_w s_l + mu_l s_w) / (s_w + s_l)). None if either class has too few confident pixels."""
    ok = np.isfinite(db)
    w, l = db[ok & (prob >= hi)], db[ok & (prob <= lo)]
    if w.size < min_px or l.size < min_px:
        return None
    sw, sl = w.std(), l.std()
    return float((w.mean() * sl + l.mean() * sw) / (sw + sl))
