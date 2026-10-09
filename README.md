# sdmad-opera-waterresource

Surface-water mapping from Sentinel-1 for **Angat Dam and reservoir** (Bulacan, Philippines): refined-Lee speckle
filtering, a robust **edge-Otsu + binary-Otsu** threshold, and a **TerraMind** foundation-model stage that learns from
the Otsu maps and settles the pixels the thresholds are unsure about.

```
Planetary Computer sentinel-1-rtc (VV, VH, linear gamma0, 10 m)      acquire.py
   -> one relative orbit, whole AOI in one frame, local solar day
   -> refined Lee (linear power)                                     speckle.py
   -> dB                                                             preprocess.py
   -> edge-Otsu  +  binary-Otsu (tiled, bimodal tiles only)          otsu.py
   -> classes: water / land / uncertain / nodata  (+ thresholds.json) pipeline.py
   -> pseudo-labels (uncertain = ignored)                            pseudo_labels.py
   -> TerraMind S1RTC encoder + light decoder, fine-tuned            terramind_seg.py
   -> fuse(model, Otsu): boundary from Otsu, ambiguity from model, disagreements flagged
```

## Quick start

```bash
pip install -r requirements.txt            # + requirements-fm.txt for the TerraMind stage
python -m pytest -q                         # offline, synthetic scenes
python scripts/run_angat.py --demo          # one synthetic scene, no network
export PC_SDK_SUBSCRIPTION_KEY=...          # optional: higher Planetary Computer rate limits
python scripts/run_angat.py --start 2024-01-01 --end 2024-12-31 --out outputs/angat
python scripts/train_terramind.py --out outputs/angat --backbone terramind_v1_base
```

## Design notes

**Acquisition** mirrors `Leads-DigiSaka-System/ai-irrigation` (`stac_data.py`, `sar.py`): `sentinel-1-rtc` via
`pystac-client`, `odc-stac` onto an explicit geobox, URLs signed at read time, retry/back-off on rate limits, STAC
results cached on disk, one relative orbit + pass direction so geometry is constant. Differences: it is read at the
native **10 m** (the irrigation project averages to 100 m, which would erase a reservoir shoreline), and an orbit day
counts only if one pass covers >= 90 % of the AOI, so one threshold serves the whole reservoir.

**Speckle** (`speckle.py`): `refined_lee` (Lee 1981: 7x7 window, 3x3 sub-windows, 8 half-window directions),
`lee`, and `multitemporal_mean`. Filtering is always in linear power; dB comes afterwards.

**Thresholds** (`otsu.py`)
- *binary-Otsu*: tile by tile, keeping only tiles that are clearly bimodal (Ashman's D, between-class variance, water
  share), median of the tile thresholds. Global Otsu is only a fallback.
- *edge-Otsu* (Donchyts et al. 2016): Canny edges, buffered, Otsu on the pixels inside the buffer. Works when water is a
  small part of the scene.
- Both must pass a **contrast gate** (classes >= 4 dB apart). Without it, edge-Otsu locks onto land-texture edges in
  scenes with little water (observed in synthetic tests: IoU collapsed from 0.99 to 0.05).
- *Combination*: if the two thresholds are within 3 dB they bracket an **uncertain band**; band pixels become water only
  when connected to pixels both methods call water (hysteresis). If they disagree by more, the edge threshold is used
  alone. If neither is valid the scene is reported as having no water instead of splitting the land histogram.

**TerraMind** (`terramind_seg.py`): the `S1RTC` modality, because Planetary Computer serves terrain-corrected gamma0.
Its normalisation differs from S1GRD (mean -10.93/-17.33 dB, std 4.39/4.46 vs -12.60/-20.29 and 5.20/5.89), taken from
terratorch 1.2.13. The ViT works on 16 px tokens, so the decoder also gets a small convolutional stem on the pixels.
The encoder is frozen by default (`--unfreeze-last N` trains the last blocks). Pseudo-labels exclude the uncertain
band, and scenes are used for training only when both Otsu methods agreed. Validation is a spatial hold-out (east strip).

`fuse` keeps Otsu's boundary where it is certain, lets the model decide the uncertain band, and **flags** pixels where
the two disagree strongly (`fm_only` = candidate flooded vegetation/shallow water, `otsu_only`) rather than silently
overriding either.

## Verification status

| Part | Status |
|---|---|
| Refined Lee, edge/binary Otsu, combination, classes, pseudo-labels | Tested on synthetic speckled scenes (`tests/`): speckle CV 0.50 -> 0.22, IoU >= 0.99 for 2-30 % water, no false lake on a land-only scene |
| TerraMind decoder, training loop, sliding-window inference, fusion, training script | Run end to end with a tiny **randomly initialised** model on synthetic data |
| Planetary Computer acquisition (`acquire.load_s1`, `run_area`) | **Not run against real data**: the build sandbox blocks Planetary Computer. Orbit choice and solar-day logic are unit-tested |
| Pretrained TerraMind weights, real Angat accuracy | **Not run**: Hugging Face was blocked. No accuracy on real Angat scenes has been measured |

## Before trusting results on Angat

- `ANGAT_BBOX` (121.08-121.26 E, 14.85-15.05 N) is an approximate box around the dam and reservoir; pass `--bbox` or edit it.
- Validate against a reference: Sentinel-2 NDWI/MNDWI from a near-coincident clear date, and/or the OPERA DSWx product.
- Radar shadow and layover on the steep slopes around the reservoir give false water. Add a HAND/slope mask (e.g. from
  Copernicus DEM) before computing areas; it is not included yet.
- `terramind_seg.refine_threshold` is an experimental heuristic and depends on how good the model is.
