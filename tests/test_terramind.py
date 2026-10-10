"""TerraMind stage on a tiny, randomly initialised model (no Hugging Face download). Skipped without torch/terratorch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

from ai_water import otsu, pipeline, pseudo_labels as pl, synthetic as syn, terramind_seg as tm  # noqa: E402


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
    from ai_water import infer
    s = syn.make_scene(shape=(256, 256), water_frac=0.15, seed=4)
    cfg = pipeline.WaterConfig(min_area_px=50, fill_holes_px=50)
    base = infer.infer_scene(s["vv"], s["vh"], None, cfg)
    assert base.prob is None and (base.fused_water == base.otsu_water).all()
    res = infer.infer_scene(s["vv"], s["vh"], model, cfg, tile=128, overlap=32, device="cpu")
    assert res.prob.shape == s["vv"].shape and res.flags.shape == s["vv"].shape
    a = res.area_km2(10.0)
    assert a["otsu"] > 0 and set(a) == {"otsu", "fused", "model_only_candidates", "otsu_only_candidates"}


def test_training_accepts_float16_patches(model):
    s = syn.make_scene(shape=(256, 256), water_frac=0.2, seed=1)
    wm = pipeline.map_water(s["vv"], s["vh"], pipeline.WaterConfig(min_area_px=30, fill_holes_px=30))
    X, Y, C = pl.extract_patches(pl.normalise_s1(wm.db, wm.vh_db), pl.labels_from_classes(wm.classes), size=64, stride=64)
    X16 = X.astype(np.float16)
    _, hist = tm.train(model, X16, Y, list(range(len(X) - 2)), [len(X) - 2, len(X) - 1], epochs=2, batch=4, lr=1e-3, device="cpu")
    assert np.isfinite(hist[-1]["loss"]) and np.isfinite(hist[-1]["val_iou"])


# ------------------------------------------------------------------ small (trained-weights-only) checkpoints
def _tiny(seed=0, **kw):
    torch.manual_seed(seed)
    return tm.TerraMindWater("terramind_v1_tiny", pretrained=False, width=32, **kw)


def test_trainable_only_checkpoint_holds_just_what_training_changed(tmp_path):
    m = _tiny()
    full, light = tmp_path / "full.pt", tmp_path / "light.pt"
    tm.save_checkpoint(m, str(full), {})
    tm.save_checkpoint(m, str(light), {}, trainable_only=True)
    keys = torch.load(str(light), weights_only=True)["state_dict"].keys()
    assert keys and not any(k.startswith("enc.") for k in keys)                       # frozen encoder is not stored
    assert any("running_mean" in k for k in keys)                                      # BatchNorm statistics are
    assert light.stat().st_size < 0.5 * full.stat().st_size


def test_unfrozen_encoder_blocks_are_saved_too(tmp_path):
    m = _tiny(unfreeze_last=1)
    tm.save_checkpoint(m, str(tmp_path / "l.pt"), {}, trainable_only=True)
    enc = [k for k in torch.load(str(tmp_path / "l.pt"), weights_only=True)["state_dict"] if k.startswith("enc.")]
    all_enc = [k for k in m.state_dict() if k.startswith("enc.")]
    assert enc and len(enc) < len(all_enc)


def test_small_checkpoint_reproduces_the_trained_model_exactly(tmp_path):
    m = _tiny(seed=0)
    with torch.no_grad():                                             # pretend training changed the decoder
        for p in list(m.head.parameters()) + list(m.proj.parameters()):
            p.add_(0.05 * torch.randn_like(p))
        m.head[1].running_mean.add_(0.3)
    p = tmp_path / "light.pt"
    tm.save_checkpoint(m, str(p), {"val_iou": 0.93}, trainable_only=True)
    x = torch.randn(2, 2, 64, 64)
    m.eval()
    with torch.no_grad():
        ref = m(x)
    torch.manual_seed(0)                    # same random encoder as `m`: stands in for "the pretrained encoder from Hugging Face"
    m2, meta = tm.load_checkpoint(str(p), pretrained_backbone=False)
    with torch.no_grad():
        assert torch.allclose(m2(x), ref, atol=1e-6)
    assert meta == {"val_iou": 0.93}


def test_small_checkpoint_that_does_not_fit_is_refused(tmp_path):
    m = _tiny()
    p = tmp_path / "light.pt"
    tm.save_checkpoint(m, str(p), {}, trainable_only=True)
    ck = torch.load(str(p), weights_only=True)
    ck["state_dict"]["head.bogus.weight"] = torch.zeros(1)
    torch.save(ck, str(p))
    with pytest.raises(RuntimeError, match="does not match"):
        tm.load_checkpoint(str(p), pretrained_backbone=False)


def test_checkpoints_load_with_weights_only_even_if_meta_had_numpy_or_paths(tmp_path):
    from pathlib import Path
    p = tmp_path / "m.pt"
    tm.save_checkpoint(_tiny(), str(p), {"iou": np.float32(0.5), "n": np.int64(7), "where": Path("/x/y")})
    ck = torch.load(str(p), weights_only=True)                          # no pickled objects inside
    assert ck["meta"]["where"] == "/x/y" and float(ck["meta"]["iou"]) == 0.5
