"""TerraMind stage on a tiny, randomly initialised model (no Hugging Face download). Skipped without torch/terratorch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

from ai_water import features as ft, labels as lb, scene, synthetic as syn, terramind_seg as tm  # noqa: E402

C = ft.N_CHANNELS


def scene_data(seed, orbit, size=64, shape=(256, 256)):
    """(patches X (n, C, s, s), Y) of a synthetic valley scene with JRC-style labels."""
    s = syn.make_terrain_scene(shape, seed=seed, orbit=orbit)
    f = scene.prepare(s["vv"], s["vh"])
    t = ft.terrain_from_dem(s["dem"], 10.0)
    lab, _, _ = lb.jrc_labels(syn.jrc_month_from_truth(s["truth"]), f.vv_db, t.slope)
    x = ft.compose(f.vv_db, f.vh_db, f.sdwi, t, orbit)
    cs = ft.extract_patches(x, lab, size, size, 0.5)
    return np.stack([x[:, r:r + size, c:c + size] for r, c in cs]), np.stack([lab[r:r + size, c:c + size] for r, c in cs]), s, x


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tm.TerraMindWater("terramind_v1_tiny", pretrained=False, freeze_backbone=True, width=32)


def test_forward_shape_and_backbone_frozen(model):
    out = model(torch.randn(2, C, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert not any(p.requires_grad for p in model.enc.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_the_encoder_reads_only_vv_and_vh_but_the_decoder_sees_every_channel(model):
    model.eval()
    x = torch.randn(1, C, 64, 64)
    y = x.clone()
    y[:, ft.N_ENCODER:] = 0.0                                  # change everything except VV, VH
    with torch.no_grad():
        assert not torch.allclose(model(x), model(y), atol=1e-4)               # terrain / SDWI / land cover reach the output
        f1, f2 = model.enc({tm.MODALITY: x[:, :ft.N_ENCODER]}), model.enc({tm.MODALITY: y[:, :ft.N_ENCODER]})
    assert all(torch.equal(a, b) for a, b in zip(f1, f2))                      # and the pretrained encoder never saw them


def test_input_must_be_multiple_of_16_and_have_the_right_channels(model):
    with pytest.raises(ValueError, match="multiple of 16"):
        model(torch.randn(1, C, 60, 64))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 2, 64, 64))


def test_predict_scene_covers_non_multiple_sizes(model):
    prob = tm.predict_scene(model, np.zeros((C, 100, 130), np.float32), tile=64, overlap=16, device="cpu")
    assert prob.shape == (100, 130) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1


def test_training_reduces_loss_and_runs_the_land_cover_dropout(model):
    X, Y, _, _ = scene_data(0, "descending")
    _, hist = tm.train(model, tm.ArrayPatches(X, Y), list(range(len(X))), [], epochs=4, batch=8, lr=2e-3, device="cpu",
                       context_dropout=1.0)
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_training_accepts_float16_patches(model):
    X, Y, _, _ = scene_data(1, "ascending")
    X16 = X.astype(np.float16)
    _, hist = tm.train(model, tm.ArrayPatches(X16, Y), list(range(len(X) - 2)), [len(X) - 2, len(X) - 1], epochs=2, batch=4,
                       lr=1e-3, device="cpu")
    assert np.isfinite(hist[-1]["loss"]) and np.isfinite(hist[-1]["val_iou"])


def test_with_terrain_the_model_learns_that_dark_shadow_is_not_water():
    """VV alone cannot tell the lake (-20 dB) from shadow (-22 dB). The terrain channels can; JRC-style labels teach it."""
    torch.manual_seed(1)
    m = tm.TerraMindWater("terramind_v1_tiny", pretrained=False, width=32)
    parts = [scene_data(seed, orbit) for seed in (0, 1, 2) for orbit in ("descending", "ascending")]
    X, Y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    tm.train(m, tm.ArrayPatches(X, Y), list(range(len(X))), [], epochs=12, batch=8, lr=3e-3, device="cpu", context_dropout=0.0)
    for orbit in ("descending", "ascending"):                                      # a new valley, both geometries
        _, _, s, x = scene_data(7, orbit)
        p = tm.predict_scene(m, x, tile=64, overlap=16, device="cpu")
        lake, shadow = p[s["truth"]].mean(), p[s["shadow"]].mean()
        assert lake > 0.6 and shadow < 0.3, (orbit, lake, shadow)


def test_terrain_rule_removes_water_predicted_on_steep_ground(model, monkeypatch):
    from ai_water import infer
    s = syn.make_terrain_scene((128, 128), seed=2)
    t = ft.terrain_from_dem(s["dem"], 10.0)
    monkeypatch.setattr(tm.TerraMindWater, "predict_prob", lambda self, x, **kw: np.full(x.shape[1:], 0.9, np.float32))
    res = infer.infer_scene(s["vv"], s["vh"], "descending", t, model)
    steep = t.slope > infer.MAX_SLOPE_DEG
    assert steep.any() and (res.flags[steep] == infer.FLAG_TERRAIN).all() and not res.water[steep].any()
    assert res.water[~steep & res.valid].all() and res.orbit == "descending"
    a = res.area_km2(10.0)
    assert a["terrain_suppressed"] > 0 and a["water"] > 0
    off = infer.infer_scene(s["vv"], s["vh"], "descending", t, model, max_slope_deg=None)
    assert off.water[steep & off.valid].all() and not (off.flags == infer.FLAG_TERRAIN).any()


# ------------------------------------------------------------------ checkpoints and inference
def test_checkpoint_roundtrip_rebuilds_without_pretrained_download(model, tmp_path):
    p = tmp_path / "m.pt"
    tm.save_checkpoint(model, str(p), {"sites": ["angat"]})
    m2, meta = tm.load_checkpoint(str(p))
    assert meta == {"sites": ["angat"]} and m2.backbone_name == "terramind_v1_tiny" and not m2.training and m2.in_channels == C
    x = torch.randn(1, C, 64, 64)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), m2(x), atol=1e-5)


def test_a_checkpoint_from_the_old_two_channel_design_is_refused(model, tmp_path):
    p = tmp_path / "old.pt"
    tm.save_checkpoint(model, str(p), {})
    ck = torch.load(str(p), weights_only=True)
    del ck["channels"]
    torch.save(ck, str(p))
    with pytest.raises(RuntimeError, match="retrain"):
        tm.load_checkpoint(str(p))


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
    x = torch.randn(2, C, 64, 64)
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
