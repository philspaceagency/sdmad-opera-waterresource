"""Output file names of the product.

    philsa_multimodal-surfacewater-weekly_YYYYMMDD_v0.1.0_r01_surfacewater.tif   water mask of one Sentinel-1 date (0 land, 1 water)
    philsa_multimodal-surfacewater-weekly_YYYYMMDD_v0.1.0_r01_flags.tif          0 land, 1 water, 2 removed by the terrain safeguard
    philsa_multimodal-surfacewater-weekly_YYYYMMDD_v0.1.0_r01_preview.png        colour picture of the two above
    philsa_multimodal-surfacewater-weekly_YYYYMMDD_v0.1.0_r01_surfacewater.gpkg  the water mask as polygons (field `water`: 1 water, 0 non-water)
    philsa_multimodal-surfacewaterfreq-frequency_YYYYMM-YYYYMM_v0.1.0_r01_surfacewaterfreq.png   water frequency over the months covered

YYYYMMDD is the local solar day of the pass. The names carry no site or pass direction: each site has its own folder, and two passes
on the same day would share a name (the later one replaces the earlier; the run logs a warning). `prob` (optional) is written as
`..._surfacewaterprob.tif`.
"""
import pandas as pd

PREFIX = "philsa_multimodal"
PRODUCT_VERSION = "0.1.0"
REVISION = "r01"
SCENE_KINDS = {"water": ("surfacewater", "tif"), "flags": ("flags", "tif"), "preview": ("preview", "png"), "prob": ("surfacewaterprob", "tif"),
               "gpkg": ("surfacewater", "gpkg")}


def scene_name(day, output, version=PRODUCT_VERSION, revision=REVISION):
    """File name of `output` ('water' | 'flags' | 'preview' | 'prob' | 'gpkg') for the pass of local day `day`."""
    kind, ext = SCENE_KINDS[output]
    return f"{PREFIX}-surfacewater-weekly_{pd.Timestamp(str(day)).strftime('%Y%m%d')}_v{version}_{revision}_{kind}.{ext}"


def frequency_name(first_day, last_day, version=PRODUCT_VERSION, revision=REVISION):
    """File name of the frequency picture for passes from `first_day` to `last_day` (months, inclusive)."""
    a, b = (pd.Timestamp(str(d)).strftime("%Y%m") for d in (first_day, last_day))
    return f"{PREFIX}-surfacewaterfreq-frequency_{a}-{b}_v{version}_{revision}_surfacewaterfreq.png"
