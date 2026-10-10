# sdmad-opera-waterresource

Surface-water mapping from Sentinel-1 for **Angat Dam, Magat Dam and other water bodies** in the Philippines.

1. **Otsu water maps** - refined-Lee speckle filtering and a robust **edge-Otsu + binary-Otsu** threshold.
2. **A labelled training dataset** - each Sentinel-1 date labelled with the **JRC Monthly History** map of the *same month*
   (Google Earth Engine), kept only where JRC and Otsu agree.
3. **TerraMind** (`S1RTC`) fine-tuned on that dataset, then **fused** with Otsu for inference on any date.

```
Planetary Computer sentinel-1-rtc (VV, VH, linear gamma0, 10 m)      acquire.py
   -> one relative orbit, whole site in one frame, local solar day
   -> refined Lee (linear power) -> dB                               speckle.py, preprocess.py
   -> edge-Otsu + binary-Otsu -> water / land / uncertain / nodata   otsu.py, pipeline.py
   -> + JRC monthly map of the SAME calendar month (Earth Engine)    jrc.py
   -> tiered labels + per-pixel source + JRC-Otsu agreement          labels.py, dataset.py
   -> TerraMind S1RTC encoder + light decoder, trained               terramind_seg.py
   -> inference: Otsu + P(water) -> fused map + disagreement flags   infer.py
```

## Quick start

```bash
pip install -r requirements.txt            # core, acquisition, Earth Engine
pip install -r requirements-fm.txt         # + torch and terratorch for TerraMind
python -m pytest -q                         # offline, synthetic data (torch tests skip without torch)
```

**Notebook (recommended):** `notebooks/angat_water_train_infer.ipynb` runs on **real data**: dataset (Planetary Computer + Earth
Engine) -> TerraMind training (pretrained, Hugging Face) -> inference -> time series -> export. Set `GEE_PROJECT` in the config
cell, use a **GPU runtime**, and expect the dataset step to be the long one (about 125 passes per site for 2018-2021, network-bound).
It is **resumable**: every scene is appended to `manifest.jsonl` on Drive as it finishes, and re-running the dataset cell after a
disconnect skips the passes already done without reading them again (`RESTART = True` starts over). Patches are held in RAM as
float16 and capped per scene (`MAX_PATCHES_PER_SCENE`), so the full dataset fits in Colab memory.

**Google Colab:** the notebook's setup cell installs the dependencies itself when it detects Colab. It pins `numpy<2.3`, because Colab's
preinstalled `numba` requires it; without the pin pip upgrades numpy and prints `numba requires numpy<2.3`. If you see that message,
or numpy was imported before the install, use Runtime > Restart session and run again. The repository root is hard-coded in that cell
(`ROOT = Path("/content/drive/MyDrive/Surface_Water/project")`, Drive is mounted if needed): edit that one line if your copy lives elsewhere. The same pin is safe elsewhere: the tests
pass on numpy 2.2.6 as well as on 2.5.

**If the dataset table cell fails** (`manifest.jsonl is empty` / `all N scenes were rejected` / `does not exist`), the message names the likely
cause. Empty means the dataset step read no Sentinel-1 scene: look for `unreadable` lines or a Planetary Computer token / rate-limit error in
that cell's output (set `PC_SDK_SUBSCRIPTION_KEY`), and check the dates and site box. All rejected means JRC and Otsu did not agree on any scene
(`low_agreement`: try `MAX_MONTH_OFFSET = 1`), or JRC had no coverage (`no_jrc_month`, `jrc_nodata`); the table above the error shows each scene's reason.

**Scripts:**

```bash
export PC_SDK_SUBSCRIPTION_KEY=...          # optional: higher Planetary Computer rate limits

# Otsu only, one site, any dates -> GeoTIFFs + thresholds.json
python scripts/run_angat.py --start 2024-01-01 --end 2024-12-31 --out outputs/angat
python scripts/run_angat.py --demo          # one synthetic scene, no network

# JRC-labelled, multi-site dataset (2017-2021) and training with a whole site held out
python scripts/build_dataset.py --sites angat magat --start 2018-01-01 --end 2021-12-31 --gee-project <project> --out outputs/dataset
python scripts/train_terramind.py --dataset outputs/dataset --holdout-sites magat --out outputs/model

# Otsu pseudo-labels of one site instead of JRC (no Earth Engine needed)
python scripts/train_terramind.py --out outputs/angat --backbone terramind_v1_base
```

## Repository layout

| Path | Purpose |
|---|---|
| `angat_water/acquire.py` | Planetary Computer `sentinel-1-rtc`: STAC search + cache, signing, orbit choice, 10 m reads, `iter_scenes` |
| `angat_water/speckle.py` | `refined_lee`, `lee`, `multitemporal_mean` (linear power) |
| `angat_water/preprocess.py` | linear <-> dB, border-noise masking |
| `angat_water/otsu.py` | binary-Otsu (tiled), edge-Otsu, their combination, classes, mask cleaning |
| `angat_water/pipeline.py` | `WaterConfig`, `map_water` (one scene), `run_area` (search -> per-date maps) |
| `angat_water/jrc.py` | JRC GSW from Earth Engine onto the S1 grid; same-month matching (`matched_month`) |
| `angat_water/labels.py` | tiered JRC x Otsu labels, label source, scene agreement and quality |
| `angat_water/dataset.py` | build per-scene `.npz` + `manifest.jsonl`; `load_patches`; site / block splits |
| `angat_water/sites.py` | site registry (`angat`, `magat`, `custom(...)` for rivers etc.) |
| `angat_water/pseudo_labels.py` | Otsu-only labels, patch extraction, TerraMind input normalisation |
| `angat_water/terramind_seg.py` | `TerraMindWater` model, training, sliding-window prediction, `fuse`, checkpoints |
| `angat_water/infer.py` | `infer_scene`, `iter_window` (any dates), `write_result` (GeoTIFFs), areas |
| `angat_water/synthetic.py` | synthetic speckled scenes and a synthetic JRC for the tests |
| `scripts/` | `run_angat.py`, `build_dataset.py`, `train_terramind.py` |
| `notebooks/` | `angat_water_train_infer.ipynb` |
| `tests/` | offline tests (synthetic data; mocked Earth Engine) |

## Design notes

**Acquisition** follows `Leads-DigiSaka-System/ai-irrigation` (`stac_data.py`, `sar.py`): `sentinel-1-rtc` via `pystac-client`,
`odc-stac` onto an explicit geobox, URLs signed at read time, retry/back-off on rate limits, STAC results cached on disk,
one relative orbit + pass direction so geometry is constant. Differences: it is read at the native **10 m** (the irrigation
project averages to 100 m, which would erase a reservoir shoreline), and an orbit day counts only if one pass covers >= 90 %
of the site, so one threshold serves the whole reservoir.

**Speckle**: `refined_lee` (Lee 1981: 7x7 window, 3x3 sub-windows, 8 half-window directions), `lee`, `multitemporal_mean`.
Filtering is always in linear power; dB comes afterwards.

**Thresholds**
- *binary-Otsu*: per tile, keeping only clearly bimodal tiles (Ashman's D, between-class variance, water share), median of the
  tile thresholds. Global Otsu is only a fallback.
- *edge-Otsu* (Donchyts et al. 2016): Canny edges, buffered, Otsu on the pixels inside the buffer. Works when water is a small
  part of the scene.
- Both must pass a **contrast gate** (classes >= 4 dB apart). Without it, edge-Otsu locks onto land-texture edges in scenes with
  little water (in synthetic tests IoU collapsed from 0.99 to 0.05).
- *Combination*: thresholds within 3 dB bracket an **uncertain band**; band pixels become water only when connected to pixels
  both methods call water (hysteresis). If they disagree by more, the edge threshold is used alone. If neither is valid the scene
  has no water, instead of splitting the land histogram.

## JRC-labelled dataset (date-matched)

`dataset.build_site` labels every Sentinel-1 date with the **JRC Monthly History** map (`JRC/GSW1_4/MonthlyHistory`, Landsat,
30 m) of **the same calendar month**, read from Earth Engine onto the S1 10 m UTM grid (`ee.data.computePixels`, as in
ai-irrigation's `weather.py`). JRC is monthly, so the month is the finest match possible. `max_month_offset = 0` (default) never
borrows a neighbouring month, and the offset used is stored per scene. A month with too little Landsat coverage makes the scene
`no_jrc_month` rather than falling back silently.

Labels exist only where JRC and the same-date Otsu map are consistent (`labels.jrc_otsu_labels`):

| JRC (S1 month) | Otsu (S1 date) | Label |
|---|---|---|
| water | water | water |
| not water | land | land |
| water / not water | uncertain band | JRC decides |
| water / not water | contradicts | **ignored**, counted as a conflict |
| within 3 px (30 m) of the JRC shoreline | water / land | Otsu decides (10 m; JRC is a Landsat pixel coarser) |
| no data | any | ignored |

Every labelled pixel keeps its source (`agree`, `jrc_resolved`, `otsu_edge`, `never_water`), and each scene stores its JRC-Otsu
**agreement**. A scene is used for training only if agreement >= 0.9, JRC no-data <= 40 % and it has a real water body. Rejected
scenes stay in `manifest.jsonl` with the reason (`jrc_nodata`, `low_agreement`, `few_water_px`, `no_jrc_month`). Low agreement
means the JRC month does not match the S1 date (the reservoir moved) or Otsu failed.

- JRC GSW1_4 ends in **Dec 2021** and S1 RTC starts in 2017: labelled dates are 2017-2021. The model then runs on any later date.
- `never_water_negatives` (off) relabels "JRC never water but S1 water" as land: it removes radar-shadow errors but also teaches
  that real new floods and flooded paddies are land. Use it only where shadow is the problem.
- Add rivers or look-alike surfaces with `--site-bbox NAME W S E N --kind river` (or `sites.custom`). Rivers a few pixels wide are
  mostly mixed pixels; `load_patches` keeps only a share (`keep_empty`) of near-waterless patches so water is not drowned in land.

## Training and inference

**Model** (`terramind_seg.TerraMindWater`): TerraMind encoder with the **`S1RTC`** modality (Planetary Computer serves terrain-
corrected gamma0; its normalisation differs from S1GRD: mean -10.93/-17.33 dB, std 4.39/4.46, from terratorch 1.2.13), a
multi-layer token decoder, and a small convolutional stem on the pixels (the ViT works on 16 px tokens). The encoder is frozen by
default (`unfreeze_last = N` also trains the last blocks). Loss: cross-entropy over labelled pixels + Dice. Sizes: `terramind_v1_
tiny | small | base | large`.

**Validation**: `--holdout-sites magat` keeps whole sites out of training (the honest generalisation test); otherwise an east-
strip block of every scene is held out so overlapping patches never straddle the split. IoU is measured against the **training
labels**, so it says how well the model reproduces JRC x Otsu on unseen data, not how close it is to reality.

**Checkpoints**: `save_checkpoint` stores weights, backbone name and decoder width; `load_checkpoint` rebuilds the model without
downloading the pretrained weights again.

**Inference** (`infer.infer_scene`): Otsu classes, TerraMind P(water) by overlapping tiles, then `fuse`. Otsu keeps the 10 m
boundary where it is certain, the model decides the uncertain band, and strong disagreements are **flagged** instead of overridden:
`model-only` (candidate flooded vegetation / shallow water) and `Otsu-only`. `iter_window` reads any dates; `write_result` writes
`*_otsu_classes`, `*_fused_water`, `*_flags`, `*_prob` GeoTIFFs; `SceneResult.area_km2` gives areas for a time series.

## Saving and sharing the model

Training (notebook section 4, or `scripts/train_terramind.py`) writes two checkpoints plus the training history:

| File | Holds | Size (`terramind_v1_base`) | Where it lives |
|---|---|---|---|
| `outputs/model/terramind_water.pt` | the **complete** model, pretrained encoder included; loads offline | about 346 MB | Drive (too big for git: GitHub rejects files over 100 MB; `outputs/` and `*.pt` are git-ignored) |
| `outputs/model/terramind_water_decoder.pt` | only the weights training changed (decoder, pixel stem, head, BatchNorm statistics, plus any unfrozen encoder blocks) | about 4 MB | copied to `models/` in the repo, which `.gitignore` allows |
| `outputs/model/training_history.json` | per-epoch loss and validation IoU, plus the configuration | small | Drive |

`tm.load_checkpoint(path)` loads either. The small file rebuilds the pretrained encoder from Hugging Face and loads the trained weights on top, so it needs network
access and the **same pretrained encoder** the model was trained with (it refuses a file that does not fit the model). The notebook reloads both files right after
training and checks that they reproduce the trained model's output, then copies the small checkpoint and a model card (`models/terramind_water_decoder.json`: training
setup, validation IoU per held-out site, how to load it) into `models/`. To publish: `git add models/ && git commit && git push`. For the full 346 MB file use Git LFS, a
GitHub Release or the Hugging Face Hub instead. The validation IoU in the card is against the JRC x Otsu training labels, not against ground truth.

## Verification status

| Part | Status |
|---|---|
| Refined Lee, edge/binary Otsu, combination, classes | Tested on synthetic speckled scenes: speckle CV 0.50 -> 0.22, IoU >= 0.99 for 2-30 % water, no false lake on a land-only scene |
| JRC matching, tiered labels, dataset build/load, site hold-out, Earth Engine request blocks | Tested offline with a synthetic JRC (30 m blocks, cloud gaps, wrong-month water level) and a mocked `ee`: labels >= 99.5 % pure, radar-shadow conflicts ignored, wrong-month scenes rejected |
| TerraMind training, prediction, fusion, checkpoints, inference, export | Run end to end with a tiny **randomly initialised** model on synthetic data |
| Checkpoints (full and small), reload, exact reproduction, `.gitignore` rules | Tested: the small checkpoint holds only trained weights (0.74M of 6.2M params for `tiny`; 4.1 MB vs 346 MB for `base`), reloads to identical outputs, refuses a mismatched file, and loads with `torch.load(weights_only=True)`; `git check-ignore` confirms `models/*.pt` is trackable and `outputs/**/*.pt` is not |
| Notebook (real-data flow) | Executed top to bottom with only the four network entry points (Earth Engine init/read, Planetary Computer scene readers) replaced by stand-ins and a tiny untrained model: dataset, resume (a second run reads nothing), labelling, patches, training, checkpoint, inference, time series, export all ran. Outputs are not stored in the repository |
| **Earth Engine / JRC** (`jrc.GeeJrc`), **Planetary Computer** (`acquire.load_s1`, `iter_scenes`, `run_area`) | **Not run against the live services** from the build sandbox. Planetary Computer search and Earth Engine authentication were confirmed from your Colab run; the JRC band name was corrected after it (`water`), and the JRC reads themselves are the next thing to confirm |
| Pretrained TerraMind weights, accuracy on real Angat / Magat scenes | **Not run**: Hugging Face was blocked. No real-world accuracy has been measured |

## Before trusting results

- `ANGAT_BBOX` (121.08-121.26 E, 14.85-15.05 N) and the `magat` box (121.30-121.52 E, 16.72-16.95 N) are approximate; check them on
  a map and edit `sites.py` or pass `--site-bbox`.
- Validate against an independent reference: a clear Sentinel-2 NDWI/MNDWI date close to the pass, OPERA DSWx, or hand-checked
  patches. Watch the `agreement` column and the `model-only` / `Otsu-only` flags: they show where sources disagree.
- Radar shadow and layover on steep terrain around reservoirs give false water; a HAND/slope mask is not included yet.
- `terramind_seg.refine_threshold` is an experimental heuristic and depends on the model's quality.
