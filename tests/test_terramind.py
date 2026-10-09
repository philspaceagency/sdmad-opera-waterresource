"""TerraMind stage on a tiny, randomly initialised model (no Hugging Face download). Skipped without torch/terratorch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

from angat_water import otsu, pipeline, pseudo_labels as pl, synthetic as syn, terramind_seg as tm  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tm.TerraMindWater("terramind_v1_tiny", pretrained=False, freeze_backbone=True, width=32)


def test_forward_shape_and_backbone_frozen(model):
    out = model(torch.randn(2, 2, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert not any(p.requires_grad for p in model.enc.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_input_must_be_multiple_of_16(model):
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 60, 64))


def test_predict_scene_covers_non_multiple_sizes(model):
    prob = tm.predict_scene(model, np.full((100, 130), -10, np.float32), np.full((100, 130), -17, np.float32),
                            tile=64, overlap=16, device="cpu")
    assert prob.shape == (100, 130) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1


def test_training_reduces_loss(model):
    s = syn.make_scene(shape=(256, 256), water_frac=0.2, seed=0)
    wm = pipeline.map_water(s["vv"], s["vh"], pipeline.WaterConfig(min_area_px=30, fill_holes_px=30))
    x, y = pl.normalise_s1(wm.db, wm.vh_db), pl.labels_from_classes(wm.classes)
    X, Y, C = pl.extract_patches(x, y, size=64, stride=64, min_valid=0.9)
    _, hist = tm.train(model, X, Y, list(range(len(X))), [], epochs=4, batch=8, lr=2e-3, device="cpu")
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_fuse_semantics():
    classes = np.array([[otsu.CLASS_WATER, otsu.CLASS_WATER, otsu.CLASS_UNCERTAIN, otsu.CLASS_UNCERTAIN, otsu.CLASS_LAND]], np.uint8)
    prob = np.array([[0.8, 0.02, 0.7, 0.2, 0.99]], np.float32)
    water, flags = tm.fuse(prob, classes)
    assert water.tolist() == [[True, True, True, False, False]]
    assert flags.tolist() == [[tm.FUSE_WATER, tm.FUSE_OTSU_ONLY, tm.FUSE_WATER, tm.FUSE_LAND, tm.FUSE_FM_ONLY]]


def test_refine_threshold_between_class_means():
    rng = np.random.default_rng(0)
    db = np.r_[rng.normal(-20, 1.5, 3000), rng.normal(-9, 2.0, 3000)].astype(np.float32)
    prob = np.r_[np.full(3000, 0.99), np.full(3000, 0.01)].astype(np.float32)
    t = tm.refine_threshold(db, prob)
    assert -17 < t < -12
    assert tm.refine_threshold(db, np.full_like(prob, 0.5)) is None


# ------------------------------------------------------------------ checkpoints and inference
def test_checkpoint_roundtrip_rebuilds_without_pretrained_download(model, tmp_path):
    p = tmp_path / "m.pt"
    tm.save_checkpoint(model, str(p), {"sites": ["angat"]})
    m2, meta = tm.load_checkpoint(str(p))
    assert meta == {"sites": ["angat"]} and m2.backbone_name == "terramind_v1_tiny" and not m2.training
    x = torch.randn(1, 2, 64, 64)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), m2(x), atol=1e-5)


def test_infer_scene_with_and_without_model(model):
    from angat_water import infer
    s = syn.make_scene(shape=(256, 256), water_frac=0.15, seed=4)
    cfg = pipeline.WaterConfig(min_area_px=50, fill_holes_px=50)
    base = infer.infer_scene(s["vv"], s["vh"], None, cfg)
    assert base.prob is None and (base.fused_water == base.otsu_water).all()
    res = infer.infer_scene(s["vv"], s["vh"], model, cfg, tile=128, overlap=32, device="cpu")
    assert res.prob.shape == s["vv"].shape and res.flags.shape == s["vv"].shape
    a = res.area_km2(10.0)
    assert a["otsu"] > 0 and set(a) == {"otsu", "fused", "model_only_candidates", "otsu_only_candidates"}
