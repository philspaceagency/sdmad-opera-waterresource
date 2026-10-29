"""Output file names and the PNG pictures."""
import numpy as np
import pytest

from ai_water import batch, naming


def test_scene_names_follow_the_product_convention():
    p = "philsa_multimodal-surfacewater-weekly_20240110_v0.1.0_r01_"
    assert naming.scene_name("2024-01-10", "water") == p + "surfacewater.tif"
    assert naming.scene_name(np.datetime64("2024-01-10"), "flags") == p + "flags.tif"
    assert naming.scene_name("2024-01-10", "preview") == p + "preview.png"
    assert naming.scene_name("2024-01-10", "water", version="0.2.0", revision="r02").endswith("_v0.2.0_r02_surfacewater.tif")


def test_frequency_name_spans_the_months_covered():
    assert naming.frequency_name("2024-01-10", "2024-03-21") == \
        "philsa_multimodal-surfacewaterfreq-frequency_202401-202403_v0.1.0_r01_surfacewaterfreq.png"


def test_frequency_picture_hides_unobserved_pixels():
    f = np.array([[0, 100, 255]], np.uint8)
    rgba = batch.frequency_rgb(f)
    assert rgba.shape == (1, 3, 4) and rgba[0, 2, 3] == 0 and rgba[0, 0, 3] == 255 and not np.array_equal(rgba[0, 0], rgba[0, 1])
