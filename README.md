# ai_water

Surface-water mapping from Sentinel-1 for the Philippines: **one water body (Angat Dam by default), many bodies, or the whole country.**

1. **Otsu water maps** - refined-Lee speckle filtering and a robust **edge-Otsu + binary-Otsu** threshold.
2. **A labelled training dataset** - each Sentinel-1 date labelled with the **JRC Monthly History** map of the *same month* (Google Earth Engine), kept
   only where JRC and Otsu agree. Six water bodies are used for training: Angat, Magat and La Mesa dams, Sampaloc Lake, Libmanan and Cagayan river reaches.
3. **TerraMind** (`S1RTC`) fine-tuned on that dataset, then **fused** with Otsu for mapping on any date.

```
Planetary Computer sentinel-1-rtc (VV, VH, linear gamma0, 10 m)        acquire.py
   -> one relative orbit, the whole area in one frame, local solar day
   -> refined Lee (linear power) -> dB                                 speckle.py, preprocess.py
   -> edge-Otsu + binary-Otsu -> water / land / uncertain / nodata     otsu.py, pipeline.py

 TRAINING  (notebooks/ai_water_training.ipynb)
   + JRC monthly map of the SAME calendar month (Earth Engine)         jrc.py
   -> tiered labels + per-pixel source + JRC-Otsu agreement            labels.py, dataset.py
   -> TerraMind S1RTC encoder + light decoder -> models/               terramind_seg.py

 MAPPING   (notebooks/ai_water_inference.ipynb, scripts/map_water.py)
   area: Angat Dam (default) | a dam's vector file | many bodies | ~20 km tiles over a country   sites.py, nationwide.py
   -> Otsu + P(water) -> fused map + disagreement flags, per body and date   infer.py
   -> areas.csv, water-frequency map, per-date masks, failures.csv           batch.py
```

## Quick start

```bash
pip install -r requirements.txt            # core, acquisition, Earth Engine
pip install -r requirements-fm.txt         # + torch and terratorch for TerraMind
python -m pytest -q                         # offline, synthetic data (torch tests skip without torch)
```

Two notebooks, run in this order (both work on **real data**; use a GPU runtime):

| Notebook | What it does |
|---|---|
| `notebooks/ai_water_training.ipynb` | builds the JRC-labelled dataset for the six training sites (`data/training_sites.csv`), trains TerraMind with a site held out, saves the model (`models/terramind_water_decoder.pt` + model card) |
| `notebooks/ai_water_inference.ipynb` | maps surface water with that model. `RUN_MODE = "single"` (default **Angat Dam**, or the vector file of another dam), `"multiple"` (several bodies in one run), `"nationwide"` (the whole country in tiles) |

Set `GEE_PROJECT` in the config cell (needed for the dataset and for nationwide pre-screening). Both are **resumable**: a training dataset build or a mapping run that is
interrupted (a Colab disconnect) continues where it stopped, without reading the finished Sentinel-1 passes again.

**Google Colab:** the setup cell installs the dependencies itself when it detects Colab. It pins `numpy<2.3`, because Colab's preinstalled `numba` requires it; without the pin pip
upgrades numpy and prints `numba requires numpy<2.3`. If you see that message, or numpy was imported before the install, use Runtime > Restart session and run again. The
repository root is hard-coded in the setup cell (`ROOT = Path("/content/drive/MyDrive/Surface_Water/project")`, Drive is mounted if needed): edit that one line if your copy
lives elsewhere.

## Choosing the area to map

| You want | Set (notebook) / pass (script) |
|---|---|
| **Angat Dam** (the default) | nothing: `AOI = None` |
| another **dam, reservoir, lake or river** | `AOI = "my_dam.geojson"` / `--aoi my_dam.geojson`: a vector file with the outline. GeoJSON works as is; GeoPackage, Shapefile and KML need `geopandas`. The file must be lon/lat (GeoJSON) or carry its CRS |
| **several bodies** in one run | a file with one feature per body (or a sites CSV) and `RUN_MODE = "multiple"` / `--multiple`; `AOI_NAMES` / `--names` picks some of them. A multi-feature file is never silently cut down to its first feature |
| the **whole country** | `RUN_MODE = "nationwide"` / `--nationwide`, with `BOUNDARY` / `--boundary` = a country or province outline (or leave it out to use Earth Engine's country boundary). Start with `NATIONWIDE_BBOX` / `--bbox` or `--limit-tiles` |

Each feature becomes a site: the box of its outline plus `BUFFER_KM` (1 km, so shorelines keep their edge pixels), its own UTM zone, and the outline itself, which results are
clipped to. A feature's name comes from `NAME_COL` (else `name`, `dam`, `label`, `id`); its kind (reservoir / lake / river) is guessed from the geometry unless set. Points get a
box of a radius by kind. Sites can also be listed in a CSV (`name`, `kind`, and `corners` as `((lat, lon), (lat, lon))`, or `south,west,north,east`, or `lat,lon,radius_km`).

**Nationwide** (`nationwide.py`) cuts the boundary into ~0.18 degree (~20 km) tiles on a fixed grid (a tile always has the same name and box, which makes runs resumable), each
clipped to the land plus a 1 km margin so the open sea is not mapped. `PRESCREEN` drops tiles with almost no surface water using JRC occurrence. Tiles go through the same code as
a single site, so one failing tile does not stop the others. There is **no single national raster** (at 10 m it would be about 3 x 10^9 pixels per layer): the tiles are the
product, with `index.geojson` listing every tile and its statistics; `gdalbuildvrt` makes a virtual mosaic per UTM zone if you need one. `monthly_national_area` sums the tiles'
monthly mean areas and reports how many tiles were observed, so a month with few tiles reads as an undercount, not a drop in water.

## Outputs of a mapping run

Under `outputs/maps` (the `OUT` setting):

| File | Content |
|---|---|
| `areas.csv` | one row per site and date: Otsu and fused water area (km2), the model-only / Otsu-only candidate areas, the observed area, the Otsu mode |
| `<site>/frequency.tif` | % of the valid observations on which each pixel was water (255 = never observed): permanent water 100, seasonal or flooded areas in between |
| `<site>/n_obs.tif` | number of valid observations per pixel |
| `<site>/<date>_fused_water.tif`, `_flags.tif` ... | per-date masks (`WRITE`; none by default for nationwide) |
| `failures.csv` | sites that failed (no pass covers them, a read error) with the reason |
| `index.geojson` | nationwide only: every tile with status, dates, mean / max area |

## Scripts

```bash
export PC_SDK_SUBSCRIPTION_KEY=...          # optional: higher Planetary Computer rate limits

# training data and model (the notebook does the same)
python scripts/build_dataset.py --gee-project <project> --start 2018-01-01 --end 2021-12-31      # the six sites; resumes
python scripts/train_terramind.py --dataset outputs/dataset --holdout-sites "La Mesa Dam" --out outputs/model

# mapping
python scripts/map_water.py --start 2024-01-01 --end 2024-03-31                                   # Angat Dam
python scripts/map_water.py --aoi my_dam.geojson --start ... --end ...                            # another dam
python scripts/map_water.py --aoi dams.geojson --multiple --start ... --end ...                   # many bodies
python scripts/map_water.py --nationwide --boundary philippines.geojson --prescreen --gee-project <project> \
        --bbox 120.5 14.0 121.5 15.0 --start ... --end ...                                        # a region first, then the country
python scripts/map_water.py --otsu-only ...                                                       # thresholds only, no model needed

python scripts/run_angat.py --start 2024-01-01 --end 2024-12-31 --out outputs/angat               # Otsu only, one site -> thresholds.json
```

## Repository layout

| Path | Purpose |
|---|---|
| `ai_water/acquire.py` | Planetary Computer `sentinel-1-rtc`: STAC search + cache, signing, orbit choice, 10 m reads, `iter_scenes` |
| `ai_water/speckle.py`, `preprocess.py` | `refined_lee`, `lee`, `multitemporal_mean` (linear power); linear <-> dB |
| `ai_water/otsu.py`, `pipeline.py` | binary-Otsu (tiled), edge-Otsu, their combination; `WaterConfig`, `map_water` (one scene) |
| `ai_water/sites.py` | `Site`, the default (Angat Dam), CSV / vector-file loaders, minimum size, UTM zone, `resolve_aoi`, `load_training_sites` |
| `ai_water/aoi.py` | rasterise a site's outline to clip results |
| `ai_water/batch.py` | `map_sites`: many sites, resumable, per-site failure isolation, `areas.csv`, frequency composite |
| `ai_water/nationwide.py` | country boundary, tiles, JRC pre-screen, `run_nationwide`, index, monthly national totals |
| `ai_water/jrc.py`, `labels.py`, `dataset.py` | JRC from Earth Engine, same-month matching, tiered labels, per-scene `.npz` + `manifest.jsonl`, patches, site splits |
| `ai_water/terramind_seg.py` | `TerraMindWater` model, training, sliding-window prediction, `fuse`, checkpoints |
| `ai_water/infer.py` | `load_model`, `infer_scene`, `iter_window` (any dates), `write_result` |
| `ai_water/pseudo_labels.py`, `synthetic.py` | Otsu-only labels, patches, TerraMind normalisation; synthetic scenes for the tests |
| `data/training_sites.csv` | the six training water bodies |
| `models/` | the small trained model and its model card (see below) |
| `scripts/` | `build_dataset.py`, `train_terramind.py`, `map_water.py`, `run_angat.py` |
| `notebooks/` | `ai_water_training.ipynb`, `ai_water_inference.ipynb` |
| `tests/` | offline tests (synthetic data; Earth Engine and Planetary Computer replaced by stand-ins) |

## Training sites

`data/training_sites.csv`, with the corner coordinates as `(lat, lon)` pairs:

| Site | Kind | Box |
|---|---|---|
| Angat Dam | reservoir | 12.8 x 20.0 km |
| Magat Dam | reservoir | 22.7 x 15.6 km |
| La Mesa Dam | reservoir | 8.4 x 7.7 km |
| Sampaloc Lake | lake | 1.5 x 1.4 km, **expanded to 5 x 5 km** |
| Libmanan River | river | 5.6 x 5.9 km |
| Cagayan River | river | 11.1 x 15.1 km |

Boxes smaller than 5 km on a side are expanded about their centre (and the note says so): the model looks at 2.2 km patches and needs context, and a 1.4 km box would give no
training patches at all. One site (La Mesa Dam by default, `HOLDOUT`) is held out of training and used for validation: the honest test of how the model carries over to a water body it
has never seen. Add water bodies by adding rows. The rivers are the hardest case: a river a few pixels wide is mostly mixed pixels at 10 m.

## Design notes

**Acquisition** follows `Leads-DigiSaka-System/ai-irrigation` (`stac_data.py`, `sar.py`): `sentinel-1-rtc` via `pystac-client`, `odc-stac` onto an explicit geobox, URLs signed at read
time, retry/back-off on rate limits, STAC results cached on disk, one relative orbit + pass direction so geometry is constant. Differences: it is read at the native **10 m** (the
irrigation project averages to 100 m, which would erase a reservoir shoreline), and an orbit day counts only if one pass covers >= 90 % of the area, so one threshold serves the
whole water body.

**Speckle**: `refined_lee` (Lee 1981: 7x7 window, 3x3 sub-windows, 8 half-window directions), `lee`, `multitemporal_mean`. Filtering is always in linear power; dB comes afterwards.

**Thresholds**
- *binary-Otsu*: per tile, keeping only clearly bimodal tiles (Ashman's D, between-class variance, water share), median of the tile thresholds. Global Otsu is only a fallback.
- *edge-Otsu* (Donchyts et al. 2016): Canny edges, buffered, Otsu on the pixels inside the buffer. Works when water is a small part of the scene.
- Both must pass a **contrast gate** (classes >= 4 dB apart). Without it, edge-Otsu locks onto land-texture edges in scenes with little water (in synthetic tests IoU collapsed from
  0.99 to 0.05).
- *Combination*: thresholds within 3 dB bracket an **uncertain band**; band pixels become water only when connected to pixels both methods call water (hysteresis). If they disagree
  by more, the edge threshold is used alone. If neither is valid the scene has no water, instead of splitting the land histogram.

## JRC-labelled dataset (date-matched)

`dataset.build_site` labels every Sentinel-1 date with the **JRC Monthly History** map (`JRC/GSW1_4/MonthlyHistory`, band `water`, Landsat, 30 m) of **the same calendar month**, read
from Earth Engine onto the S1 10 m UTM grid (`ee.data.computePixels`, as in ai-irrigation's `weather.py`). JRC is monthly, so the month is the finest match possible.
`max_month_offset = 0` (default) never borrows a neighbouring month, and the offset used is stored per scene. A month with too little Landsat coverage makes the scene `no_jrc_month`
rather than falling back silently.

Labels exist only where JRC and the same-date Otsu map are consistent (`labels.jrc_otsu_labels`):

| JRC (S1 month) | Otsu (S1 date) | Label |
|---|---|---|
| water | water | water |
| not water | land | land |
| water / not water | uncertain band | JRC decides |
| water / not water | contradicts | **ignored**, counted as a conflict |
| within 3 px (30 m) of the JRC shoreline | water / land | Otsu decides (10 m; JRC is a Landsat pixel coarser) |
| no data | any | ignored |

Every labelled pixel keeps its source (`agree`, `jrc_resolved`, `otsu_edge`, `never_water`), and each scene stores its JRC-Otsu **agreement**. A scene is used for training only if
agreement >= 0.9, JRC no-data <= 40 % and it has a real water body. Rejected scenes stay in `manifest.jsonl` with the reason (`jrc_nodata`, `low_agreement`, `few_water_px`,
`no_jrc_month`). Low agreement means the JRC month does not match the S1 date (the reservoir moved) or Otsu failed.

- JRC GSW1_4 ends in **Dec 2021** and S1 RTC starts in 2017: labelled dates are 2017-2021. The model then maps any later date.
- `never_water_negatives` (off) relabels "JRC never water but S1 water" as land: it removes radar-shadow errors but also teaches that real new floods and flooded paddies are land.
- `load_patches` keeps only a share (`keep_empty`) of near-waterless patches so water is not drowned in land, caps the patches per scene and stores them as float16, so the full
  dataset fits in Colab memory.

**If the dataset table cell fails** (`manifest.jsonl is empty` / `all N scenes were rejected` / `does not exist`), the message names the likely cause. Empty means the dataset step read
no Sentinel-1 scene: look for `unreadable` lines or a Planetary Computer token / rate-limit error in the output (set `PC_SDK_SUBSCRIPTION_KEY`), and check the dates and the site box.
All rejected means JRC and Otsu did not agree on any scene (`low_agreement`: try `MAX_MONTH_OFFSET = 1`), or JRC had no coverage (`no_jrc_month`, `jrc_nodata`).

## Model, validation and mapping

**Model** (`terramind_seg.TerraMindWater`): TerraMind encoder with the **`S1RTC`** modality (Planetary Computer serves terrain-corrected gamma0; its normalisation differs from S1GRD: mean
-10.93/-17.33 dB, std 4.39/4.46, from terratorch 1.2.13), a multi-layer token decoder, and a small convolutional stem on the pixels (the ViT works on 16 px tokens). The encoder is frozen by
default (`UNFREEZE_LAST = N` also trains the last blocks). Loss: cross-entropy over labelled pixels + Dice. Sizes: `terramind_v1_tiny | small | base | large`.

**Validation**: `HOLDOUT` keeps whole sites out of training; without it an east-strip block of every scene is held out so overlapping patches never straddle the split. IoU is measured
against the **training labels**, so it says how well the model reproduces JRC x Otsu on unseen data, not how close it is to reality.

**Mapping** (`infer.infer_scene`): Otsu classes, TerraMind P(water) by overlapping tiles, then `fuse`. Otsu keeps the 10 m boundary where it is certain, the model decides the uncertain band,
and strong disagreements are **flagged** instead of overridden: `model-only` (candidate flooded vegetation / shallow water) and `Otsu-only`. Areas are counted inside the water body's outline.

## Saving and sharing the model

Training writes two checkpoints plus the history:

| File | Holds | Size (`terramind_v1_base`) | Where it lives |
|---|---|---|---|
| `outputs/model/terramind_water.pt` | the **complete** model, pretrained encoder included; loads offline | about 346 MB | Drive (too big for git: GitHub rejects files over 100 MB; `outputs/` and `*.pt` are git-ignored) |
| `outputs/model/terramind_water_decoder.pt` | only the weights training changed (decoder, pixel stem, head, BatchNorm statistics, plus any unfrozen encoder blocks) | about 4 MB | copied to `models/`, which `.gitignore` allows |
| `outputs/model/training_history.json` | per-epoch loss and validation IoU, plus the configuration | small | Drive |

`infer.load_model(path)` (or `terramind_seg.load_checkpoint`) loads either. The small file rebuilds the pretrained encoder from Hugging Face and puts the trained weights on top, so it needs
network access and the **same pretrained encoder** the model was trained with (it refuses a file that does not fit). The training notebook reloads both files and checks that they reproduce
the trained model's output, then copies the small checkpoint and a model card (`models/terramind_water_decoder.json`) into `models/`. To publish: `git add models/ && git commit && git push`.
For the full 346 MB file use Git LFS, a GitHub Release or the Hugging Face Hub.

## Verification status

| Part | Status |
|---|---|
| Refined Lee, edge/binary Otsu, combination, classes | Tested on synthetic speckled scenes: speckle CV 0.50 -> 0.22, IoU >= 0.99 for 2-30 % water, no false lake on a land-only scene |
| JRC matching, tiered labels, dataset build/load, resume, site hold-out, Earth Engine request blocks | Tested offline with a synthetic JRC (30 m blocks, cloud gaps, wrong-month water level) and a mocked `ee`: labels >= 99.5 % pure, radar-shadow conflicts ignored, wrong-month scenes rejected |
| Sites: your six boxes, corners format, minimum size, UTM zone, CSV and GeoJSON loading, `resolve_aoi`, error messages | Tested (the six sites load as supplied; Sampaloc Lake is expanded; swapped lat/lon, duplicates, a multi-feature file without `multiple`, a projected GeoJSON are all refused with a clear message). GeoPackage / Shapefile / KML loading goes through `geopandas` and is **not** tested here (it is not installed in the build environment) |
| Batch mapping: outline clipping, resume, failure isolation, frequency composite | Tested with Planetary Computer replaced by a stand-in: areas count only inside the outline, an interrupted run reads only the new passes, a failing site is recorded while the others finish, the composite equals the share of dates with water |
| Nationwide: tiles, pre-screen, run, index, monthly totals | Tested: the tiles' land clips add up exactly to the land outline, slivers are dropped, pre-screen ignores water outside the land, a run maps tiles and isolates a failing one. The **Earth Engine country boundary** (`USDOS/LSIB_SIMPLE/2017`, `country_na`) was written from memory and not run: pass your own boundary file |
| TerraMind training, prediction, fusion, checkpoints, mapping | Run end to end with a tiny **randomly initialised** model on synthetic data; both notebooks were executed top to bottom (training on your six sites with a held-out site, inference in single / single-with-vector / multiple / nationwide modes) with only the network entry points replaced by stand-ins. Outputs are not stored in the repository |
| **Earth Engine / JRC reads** (`jrc.GeeJrc`), **Planetary Computer** (`acquire.load_s1`, `iter_scenes`) | **Not run against the live services** from the build sandbox. Planetary Computer search and Earth Engine authentication were confirmed from your Colab run; the JRC band name was corrected after it (`water`), and the JRC reads themselves are the next thing to confirm |
| Pretrained TerraMind weights, accuracy on real scenes | **Not run**: Hugging Face was blocked. No real-world accuracy has been measured |

## Before trusting results

- Validate against an independent reference: a clear Sentinel-2 NDWI/MNDWI date close to the pass, OPERA DSWx, or hand-checked patches. Watch the `agreement` column and the
  `model-only` / `Otsu-only` flags: they show where sources disagree.
- Radar shadow and layover on steep terrain around reservoirs give false water; a HAND/slope mask is not included yet.
- A nationwide run is a long job (one Sentinel-1 read per tile and date): try a region first (`--bbox`), and keep `PRESCREEN` on.
- `terramind_seg.refine_threshold` is an experimental heuristic and depends on the model's quality.
