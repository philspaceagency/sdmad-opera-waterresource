# ai_water

Surface-water mapping from Sentinel-1 for the Philippines: **one water body (Angat Dam by default), many bodies, or the whole country.**

1. **Inputs** - Sentinel-1 RTC (VV, VH) from **ascending and descending** passes, the **SDWI** water index derived from them, the **DEM** with its **slope and aspect**,
   and **land cover** (ESA WorldCover) as context only.
2. **Labels** - each Sentinel-1 date labelled with the **JRC Monthly History** map of the *same month* (Google Earth Engine), corrected with the terrain. **No Otsu thresholding
   anywhere**: the model decides what is water. Six water bodies are used for training: Angat, Magat and La Mesa dams, Sampaloc Lake, Libmanan and Cagayan river reaches.
3. **TerraMind** (`S1RTC`) fine-tuned on that dataset, then used to map any date.

```
Planetary Computer sentinel-1-rtc (VV, VH, linear gamma0, 10 m)        acquire.py
   -> best ascending AND best descending relative orbit, local solar day
   -> refined Lee (linear power) -> VV dB, VH dB, SDWI                 speckle.py, preprocess.py, scene.py
 + Earth Engine: DEM -> relief, slope, aspect; WorldCover -> groups    terrain.py, features.py

 TRAINING  (notebooks/ai_water_training.ipynb)
   + JRC monthly map of the SAME calendar month (Earth Engine)         jrc.py
   -> labels: JRC, hard negatives, steep-water ignored, shoreline band labels.py, dataset.py
   -> TerraMind S1RTC encoder + decoder over 14 channels -> models/    terramind_seg.py

 MAPPING   (notebooks/ai_water_inference.ipynb, scripts/map_water.py)
   area: Angat Dam (default) | a dam's vector file | a list of bodies   sites.py
   (separate job) whole country: scripts/map_nationwide.py -> ~20 km tiles   nationwide.py
   -> P(water) per pass -> water mask (+ terrain safeguard flag), per body, date and orbit        infer.py
   -> areas.csv, water-frequency map = min(ascending, descending), per-date masks, failures.csv   batch.py
```

## Why terrain, SDWI and both pass directions

A reservoir in steep country showed white streaks of "water" all over its slopes. Those are **radar shadow**: a slope facing away from the sensor returns almost nothing, exactly
like calm water. With one fixed orbit the shadow sits on the same pixels on every date, so a frequency map shows it as permanent water. Three things fix it:

- **Both pass directions.** Shadow falls on opposite slopes in the ascending and descending geometries. The frequency map is the pixel-wise **minimum** of the two directions'
  frequencies (`frequency_ascending.tif` and `frequency_descending.tif` are written too, so you can see it).
- **Terrain channels.** The model sees relief (height above the local low, not absolute altitude), slope, aspect and the **range slope**: the slope along the radar look direction,
  positive where the ground faces away from the sensor. It can then tell a dark flat lake from a dark steep hillside.
- **Labels that contain the trap.** JRC (Landsat) says *land* on those slopes, so they are labelled land, and the ones that are dark in the radar are counted as **hard negatives** and
  kept in the training patches. (Before, labels needed Otsu and JRC to agree, so the contradictions were thrown away and the model never saw shadow as land.)

## The model's 14 input channels (`features.py`)

| Channels | What | Notes |
|---|---|---|
| 0-1 | VV dB, VH dB | the two channels the pretrained TerraMind `S1RTC` encoder reads (standardised with its own statistics) |
| 2 | **SDWI** = ln(10 x VV x VH) - 8 | Sentinel-1 Dual-Polarized Water Index (Jia et al. 2019) on the speckle-filtered linear power |
| 3-4 | relief, slope | from the Copernicus GLO-30 DEM (`COPERNICUS/DEM/GLO30`) |
| 5-6 | aspect (sin, cos) | fades to 0 on flat ground, where aspect means nothing |
| 7 | range slope | depends on the pass direction (nominal look azimuth 80 deg ascending, 280 deg descending) |
| 8-13 | land cover | one-hot: tree, shrub/grass, crop, built, bare, wetland (ESA WorldCover v200). **Water is not a class**, so the layer cannot hand over the answer |

Only channels 0-1 go through the pretrained encoder; the rest reach the decoder through the pixel stem (late fusion). TerraMind also has a native `DEM` modality; it is not used,
because it expects absolute elevation, and a multi-modal token merge cannot be checked without the pretrained weights. **Land cover is context only**: in half of the training samples
its channels are zeroed, so the model cannot lean on it, and it is the one layer you can switch off at mapping time (`USE_LANDCOVER = False`). There is no flip/rotation augmentation,
because aspect and the range slope would stop matching the image.

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
| `notebooks/ai_water_inference.ipynb` | maps surface water with that model. `RUN_MODE = "single"` (default **Angat Dam**, or the vector file of another dam), `"multiple"` (a list of bodies in one run). The whole country is a separate job: `scripts/map_nationwide.py` |

Set `GEE_PROJECT` in the config cell: Earth Engine is always needed (JRC labels in training, DEM and land cover everywhere). Both notebooks are **resumable**: a dataset build or a mapping
run that is interrupted (a Colab disconnect) continues where it stopped, without reading the finished Sentinel-1 passes again.

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
| the **whole country** | not here: `python scripts/map_nationwide.py` (below) |

Each feature becomes a site: the box of its outline plus `BUFFER_KM` (1 km, so shorelines keep their edge pixels), its own UTM zone, and the outline itself, which results are
clipped to. A feature's name comes from `NAME_COL` (else `name`, `dam`, `label`, `id`); its kind (reservoir / lake / river) is guessed from the geometry unless set. Points get a
box of a radius by kind. Sites can also be listed in a CSV (`name`, `kind`, and `corners` as `((lat, lon), (lat, lon))`, or `south,west,north,east`, or `lat,lon,radius_km`).

### Nationwide mapping (a separate script)

Individual and listed bodies are mapped in the notebook (or `scripts/map_water.py`). The whole country is a long batch job and has its own script, `scripts/map_nationwide.py`
(outputs go to `outputs/nationwide` by default). `--boundary` is a country or province outline (leave it out to use Earth Engine's country boundary); start with `--bbox` or `--limit-tiles`.

`nationwide.py` cuts the boundary into ~0.18 degree (~20 km) tiles on a fixed grid (a tile always has the same name and box, which makes runs resumable), each
clipped to the land plus a 1 km margin so the open sea is not mapped. `PRESCREEN` drops tiles with almost no surface water using JRC occurrence. Tiles go through the same code as
a single site, so one failing tile does not stop the others. There is **no single national raster** (at 10 m it would be about 3 x 10^9 pixels per layer): the tiles are the
product, with `index.geojson` listing every tile and its statistics; `gdalbuildvrt` makes a virtual mosaic per UTM zone if you need one. `monthly_national_area` sums the tiles'
monthly mean areas and reports how many tiles were observed, so a month with few tiles reads as an undercount, not a drop in water.

## Outputs of a mapping run

Under `outputs/maps` (the `OUT` setting):

| File | Content |
|---|---|
| `areas.csv` | one row per site, date and pass direction: water area (km2), the area the terrain safeguard removed, the observed area |
| `<site>/frequency.tif` | % of the valid observations on which each pixel was water (255 = never observed). With both directions: the **minimum** of the two directions' frequencies |
| `<site>/philsa_multimodal-surfacewaterfreq-frequency_YYYYMM-YYYYMM_v0.1.0_r01_surfacewaterfreq.png` | picture of that frequency (viridis, never-observed pixels transparent); the months are those of the first and last pass in the composite |
| `<site>/frequency_ascending.tif`, `frequency_descending.tif` | the per-direction frequencies, when both directions were observed |
| `<site>/n_obs.tif` | number of valid observations per pixel (both directions) |
| `<site>/philsa_multimodal-surfacewater-weekly_YYYYMMDD_v0.1.0_r01_surfacewater.tif` | water mask of one pass (0 land, 1 water); `YYYYMMDD` = local solar day. Per-date files are written with `WRITE` (`map_nationwide.py`: none unless `--write`) |
| `<site>/..._r01_flags.tif`, `..._r01_preview.png` | 0 land, 1 water, 2 the model said water but the slope is steeper than `MAX_SLOPE_DEG` (25 deg: removed, listed for review); a colour picture of the mask (blue water, orange removed, grey no data). `..._surfacewaterprob.tif` (probability) only on request |
| `failures.csv` | sites that failed (no pass covers them, a read error) with the reason |
| `index.geojson` | nationwide only: every tile with status, scenes, mean / max area |

## Scripts

```bash
export PC_SDK_SUBSCRIPTION_KEY=...          # optional: higher Planetary Computer rate limits

# training data and model (the notebook does the same)
python scripts/build_dataset.py --gee-project <project> --start 2018-01-01 --end 2021-12-31      # the six sites; resumes
python scripts/train_terramind.py --dataset outputs/dataset --holdout-sites "La Mesa Dam" --out outputs/model

# mapping (needs a trained model and Earth Engine for the terrain)
python scripts/map_water.py --gee-project <project> --start 2024-01-01 --end 2024-03-31          # Angat Dam
python scripts/map_water.py --aoi my_dam.geojson --start ... --end ...                            # another dam
python scripts/map_water.py --aoi dams.geojson --multiple --start ... --end ...                   # many bodies

# the whole country (separate script)
python scripts/map_nationwide.py --boundary philippines.geojson --prescreen --gee-project <project> \
        --bbox 120.5 14.0 121.5 15.0 --start ... --end ...                                        # a region first, then the country
```

## Repository layout

| Path | Purpose |
|---|---|
| `ai_water/acquire.py` | Planetary Computer `sentinel-1-rtc`: STAC search + cache, signing, best orbit per pass direction, 10 m reads, `iter_scenes` |
| `ai_water/speckle.py`, `preprocess.py`, `scene.py` | `refined_lee`, `lee`; linear <-> dB; one scene -> filtered VV / VH dB and SDWI, GeoTIFF writer |
| `ai_water/features.py` | the 14 model channels: SDWI, terrain derivatives, range slope, land-cover groups, patch cutting |
| `ai_water/terrain.py` | DEM and WorldCover from Earth Engine onto a site grid, cached per grid |
| `ai_water/sites.py` | `Site`, the default (Angat Dam), CSV / vector-file loaders, minimum size, UTM zone, `resolve_aoi`, `load_training_sites` |
| `ai_water/aoi.py` | rasterise a site's outline to clip results |
| `ai_water/batch.py` | `map_sites`: many sites, resumable, per-site failure isolation, `areas.csv`, per-direction frequency composite |
| `ai_water/nationwide.py` | country boundary, tiles, JRC pre-screen, `run_nationwide`, index, monthly national totals |
| `ai_water/jrc.py`, `labels.py`, `dataset.py` | JRC from Earth Engine, same-month matching, terrain-corrected labels, per-scene `.npz` + `manifest.jsonl`, `PatchSet`, site splits |
| `ai_water/terramind_seg.py` | `TerraMindWater` model, training, sliding-window prediction, checkpoints |
| `ai_water/infer.py` | `load_model`, `infer_scene`, `iter_window` (any dates), `write_result` |
| `ai_water/synthetic.py` | synthetic scenes (including a valley with radar shadow) and stand-ins for the tests |
| `data/training_sites.csv` | the six training water bodies |
| `models/` | the small trained model and its model card (see below) |
| `scripts/` | `build_dataset.py`, `train_terramind.py`, `map_water.py` (individual / listed bodies), `map_nationwide.py` |
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
has never seen. Add water bodies by adding rows; **rows on steep terrain are what teach the model about shadow**. The rivers are the hardest case: a river a few pixels wide is mostly
mixed pixels at 10 m.

## Design notes

**Acquisition** follows `Leads-DigiSaka-System/ai-irrigation` (`stac_data.py`, `sar.py`): `sentinel-1-rtc` via `pystac-client`, `odc-stac` onto an explicit geobox, URLs signed at read
time, retry/back-off on rate limits, STAC results cached on disk. Per pass direction (ascending, descending) it keeps one relative orbit, so the viewing geometry is constant within a
direction. Differences: it is read at the native **10 m** (the irrigation project averages to 100 m, which would erase a reservoir shoreline), and an orbit day counts only if one pass
covers >= 90 % of the area, so the whole water body is judged from one image.

**Speckle**: `refined_lee` (Lee 1981: 7x7 window, 3x3 sub-windows, 8 half-window directions), `lee`, `multitemporal_mean`. Filtering is always in linear power; dB and SDWI come afterwards.
VV decides which pixels are valid (border noise); VH has a lower floor (-45 dB) because water and shadow can be darker than -30 dB in the cross-polar band.

## JRC-labelled dataset (date-matched)

`dataset.build_site` labels every Sentinel-1 date with the **JRC Monthly History** map (`JRC/GSW1_4/MonthlyHistory`, band `water`, Landsat, 30 m) of **the same calendar month**, read
from Earth Engine onto the S1 10 m UTM grid (`ee.data.computePixels`, as in ai-irrigation's `weather.py`). JRC is monthly, so the month is the finest match possible.
`max_month_offset = 0` (default) never borrows a neighbouring month, and the offset used is stored per scene. A month with too little Landsat coverage makes the scene `no_jrc_month`
rather than falling back silently.

JRC is independent of the radar, so shadow and wet soil cannot leak into the labels. The radar only audits the labels (`labels.jrc_labels`):

| JRC (S1 month) | Terrain / radar | Label |
|---|---|---|
| water | slope <= 15 deg | water (`jrc_water`) |
| water | slope > 15 deg | **ignored** (`terrain_ignored`): open water is flat; usually Landsat topographic shadow |
| not water | VV >= -18 dB | land (`jrc_land`) |
| not water | VV < -18 dB | land, recorded as a **hard negative**: shadow, wet soil, smooth bare ground |
| within 3 px (30 m) of the JRC shoreline | any | ignored (JRC is a Landsat pixel coarser; the model sharpens the edge from the 10 m radar) |
| no data (JRC or radar) | any | ignored |

A scene is used for training only if it has a real water body (>= 2000 labelled water px), JRC no-data <= 40 %, and the JRC-water pixels are at least **5 dB darker** in VV than the ordinary
JRC-land pixels (hard negatives excluded, so a shadowed hillside does not hide the contrast). Low contrast means the JRC month does not match the pass (the reservoir moved) or the water
was too rough for the radar. Rejected scenes stay in `manifest.jsonl` with the reason (`jrc_nodata`, `few_water_px`, `low_contrast`, `no_jrc_month`). The thresholds (15 deg, -18 dB, 5 dB)
are starting points, not calibrated values: they are arguments of `DatasetConfig`, the notebook and `scripts/build_dataset.py`.

- JRC GSW1_4 ends in **Dec 2021** and S1 RTC starts in 2017: labelled dates are 2017-2021. The model then maps any later date.
- The labels inherit JRC's errors (Landsat clouds, 30 m, monthly): a flood that came and went inside the month is land or water in the whole month.
- `load_patches` keeps only a share (`keep_empty`) of near-waterless patches so water is not drowned in land, but keeps every patch with hard negatives; it caps the patches per scene
  and stores the three S1 channels as float16, cutting the terrain channels from the site raster per batch, so the dataset fits in Colab memory.

**If the dataset table cell fails** (`manifest.jsonl is empty` / `all N scenes were rejected` / `does not exist`), the message names the likely cause. Empty means the dataset step read
no Sentinel-1 scene: look for `unreadable` lines or a Planetary Computer token / rate-limit error in the output (set `PC_SDK_SUBSCRIPTION_KEY`), and check the dates and the site box.
All rejected means JRC and the radar did not fit on any scene (`low_contrast`: try `MAX_MONTH_OFFSET = 1`), or JRC had no coverage (`no_jrc_month`, `jrc_nodata`).

## Model, validation and mapping

**Model** (`terramind_seg.TerraMindWater`): TerraMind encoder with the **`S1RTC`** modality (Planetary Computer serves terrain-corrected gamma0; its normalisation differs from S1GRD: mean
-10.93/-17.33 dB, std 4.39/4.46, from terratorch 1.2.13), a multi-layer token decoder, and a convolutional stem on all 14 channels (the ViT works on 16 px tokens; the stem keeps shorelines
sharp and carries the terrain). The encoder is frozen by default (`UNFREEZE_LAST = N` also trains the last blocks). Loss: cross-entropy over labelled pixels + Dice. Sizes:
`terramind_v1_tiny | small | base | large`. A checkpoint records its channel list; one trained with the old 2-channel design is refused with a message to retrain.

**Validation**: `HOLDOUT` keeps whole sites out of training; without it an east-strip block of every scene is held out so overlapping patches never straddle the split. IoU is measured
against the **training labels**, so it says how well the model reproduces JRC on unseen data, not how close it is to reality.

**Mapping** (`infer.infer_scene`): VV/VH/SDWI + terrain -> P(water) by overlapping tiles -> water where P >= `THRESHOLD`. The model decides. As a last safeguard, water on ground steeper than
`MAX_SLOPE_DEG` (25 deg, well above the DEM slope at a steep shoreline) is removed and flagged `2` so it can be reviewed; set it to `None` to see the raw model. Areas are counted inside the
water body's outline, per pass direction.

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
For the full 346 MB file use Git LFS, a GitHub Release or the Hugging Face Hub. **No trained model is committed yet**: the earlier 2-channel model was never kept, and this design needs
retraining on the real data.

## Verification status

| Part | Status |
|---|---|
| Refined Lee, SDWI, DEM derivatives (slope, aspect, relief), range slope, land-cover groups, the channel stack | Tested on synthetic data and analytic cases (a tilted plane has the right slope and aspect, the radar-look slope flips sign between the two geometries, water is never a land-cover group) |
| JRC matching, labels, dataset build/load, resume, site hold-out, Earth Engine request blocks | Tested offline with a synthetic JRC (30 m blocks, cloud gaps, wrong-month water level), a synthetic valley with radar shadow, and a mocked `ee`: shadow becomes a hard negative, JRC water on steep ground is ignored, wrong-month scenes are rejected |
| Both pass directions: orbit choice, scene streaming, resume, composite | Tested: the best orbit of each direction is chosen, scenes stream in date order, the minimum of the two directions' frequencies removes the shadow that is water-like in only one |
| Sites, area selection, batch mapping, nationwide tiling/pre-screen/index | Tested as before, with Planetary Computer, Earth Engine and the model replaced by stand-ins. GeoPackage / Shapefile / KML loading goes through `geopandas` and is **not** tested here. The **Earth Engine country boundary** (`USDOS/LSIB_SIMPLE/2017`, `country_na`) was written from memory and not run: pass your own boundary file |
| TerraMind model, training, checkpoints, terrain safeguard | Run with a tiny **randomly initialised** model on synthetic data. On synthetic valleys, a model trained with the terrain channels learns that the dark shadowed flank is not water in both geometries |
| Notebooks | Executed top to bottom with only the network entry points replaced by stand-ins (see the commit message for the exact run). Outputs are not stored in the repository |
| **Earth Engine reads of the DEM and WorldCover** (`terrain.GeeTerrain`: `COPERNICUS/DEM/GLO30` band `DEM`, `ESA/WorldCover/v200` band `Map`), **JRC reads** (`jrc.GeeJrc`), **Planetary Computer** (`acquire.load_s1`, `iter_scenes`) | **Not run against the live services** from the build sandbox. The asset ids and band names were written from memory and are looked up at run time, so a mismatch fails with the list of real bands. Planetary Computer search and Earth Engine authentication were confirmed from your Colab run |
| Pretrained TerraMind weights, accuracy on real scenes | **Not run**: Hugging Face was blocked. No real-world accuracy has been measured, and whether the terrain channels remove the streaks on the real Angat map is unconfirmed until you retrain and map |

## Before trusting results

- Validate against an independent reference: a clear Sentinel-2 NDWI/MNDWI date close to the pass, OPERA DSWx, or hand-checked patches. Compare `frequency_ascending.tif` with
  `frequency_descending.tif`: a streak that is bright in only one of them is shadow.
- The hard-negative thresholds and the 25 deg safeguard are untuned defaults. A reservoir with steep banks and no steep training site will still be a weak case.
- A nationwide run is a long job (one Sentinel-1 read per tile, date and direction): try a region first (`--bbox`), and keep `PRESCREEN` on.

## Output file names

Set in `ai_water/naming.py`: `philsa_multimodal-surfacewater-weekly_YYYYMMDD_v<version>_<revision>_<kind>`. Version `0.1.0` and revision `r01` are `MapConfig.version` / `MapConfig.revision`.
The names carry no site and no pass direction (each site has its own folder), so two passes on the same local day would replace each other's files; the run logs a warning if that happens.
The per-date files are per Sentinel-1 pass, not weekly composites; "weekly" is only the product-name token.
