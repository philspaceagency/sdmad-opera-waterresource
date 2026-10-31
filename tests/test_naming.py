"""Output file names and the PNG pictures."""
import numpy as np
import pytest

from ai_water import naming


def test_scene_names_follow_the_product_convention():
    p = "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_"
    assert naming.scene_name("2024-01-10", "water") == p + "surfacewater.tif"
    assert naming.scene_name(np.datetime64("2024-01-10"), "flags") == p + "flags.tif"
    assert naming.scene_name("2024-01-10", "preview") == p + "preview.png"
    assert naming.scene_name("2024-01-10", "water", version="0.2.0", revision="r02").endswith("_v0.2.0_r02_surfacewater.tif")


def test_frequency_name_spans_the_months_covered():
    assert naming.frequency_name("2024-01-10", "2024-03-21") == \
        "philsa_multimodal-surfacewaterfreq-frequency_202401-202403_v0.1.0_r01_surfacewaterfreq.png"


def test_gpkg_name():
    assert naming.scene_name("2024-01-10", "gpkg") == "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_surfacewater.gpkg"
