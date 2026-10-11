"""Bounded scientific regression checks. Training is forbidden in this suite."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pi_aecroformer import Retriever
from pi_aecroformer.api import INPUT_ORDER, OUTPUT_ORDER, SEEDS, default_root, sample_parameters
from pi_aecroformer import _legacy


@pytest.fixture(scope="session")
def runtime():
    return Retriever(default_root())


@pytest.fixture(scope="session")
def observations(runtime):
    frame = pd.read_csv(runtime.root / "examples" / "nasa" / "paper_input_113.csv").head(2)
    return frame, frame[list(INPUT_ORDER)].to_numpy(dtype=float)


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training/GPU initialization is forbidden during smoke checks.")
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(optimizer, "step", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)


@pytest.mark.parametrize("bad", [np.ones(4), np.ones((2, 6)), [1., 1., float("nan"), 1., 1.], [1., 1., -1., 1., 1.]])
def test_invalid_inputs(runtime, bad):
    with pytest.raises(ValueError):
        runtime.retrieve(bad, return_uncertainty=False)


def test_empty_batch(runtime):
    output = runtime.retrieve(np.empty((0, 5)))
    assert all(output[name].shape == (0,) for name in OUTPUT_ORDER)


@pytest.mark.parametrize("mode", ["ORI", "UNC"])
def test_model_outputs_match_frozen_loader(runtime, observations, mode):
    _, x = observations
    # The original architecture and feature transforms live in the immutable snapshot
    # on the developer side. The published frozen primitives have the same equations.
    for seed in SEEDS:
        record = runtime.manifest["models"][mode][str(seed)]
        sx = _legacy._load_pickle(runtime.root / record["scaler_x"])
        sy = _legacy._load_pickle(runtime.root / record["scaler_y"])
        checkpoint = torch.load(runtime.root / record["checkpoint"], map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        model = _legacy.ImprovedDeepForwardModel_MH2(input_dim=5 if mode == "ORI" else 11, output_dim=4, use_uncertainty=mode == "UNC")
        model.load_state_dict(state, strict=True)
        model.eval()
        features = _legacy._build_features_11_from_x5(x)
        if mode == "ORI":
            features = features[:, :5]
        with torch.no_grad():
            result = model(torch.tensor(sx.transform(features.astype(np.float64)).astype(np.float32)))
        mean = result[0] if mode == "UNC" else result
        expected = _legacy._clip_params4(sy.inverse_transform(mean.numpy().astype(np.float64)))
        actual, raw_std, calibrated_std = runtime.predict_parameters(x, mode, seed)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        if mode == "UNC":
            expected_std = np.sqrt(np.exp(result[1].numpy().astype(np.float64))) * sy.scale_
            np.testing.assert_allclose(raw_std, expected_std, atol=1e-12, rtol=1e-12)
            temperatures = json.loads((runtime.root / record["temperature"]).read_text(encoding="utf-8"))["temperatures"]
            np.testing.assert_allclose(calibrated_std, raw_std * temperatures, atol=1e-12, rtol=1e-12)


def test_single_batch_and_ensemble_centers(runtime, observations):
    _, x = observations
    ensemble = runtime.retrieve(x, "UNC", return_uncertainty=False)
    individuals = [runtime.retrieve(x, "UNC", model_seed=seed, return_uncertainty=False) for seed in SEEDS]
    single = runtime.retrieve(x[0], "UNC", return_uncertainty=False)
    for name in OUTPUT_ORDER:
        np.testing.assert_allclose(ensemble[name], np.mean([out[name] for out in individuals], axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(single[name], ensemble[name][0], atol=1e-7, rtol=1e-6)


def test_uncertainty_propagates_n_and_v(runtime, observations):
    _, x = observations
    seed, random_seed, draws = SEEDS[0], 20261020, 4
    mean, _, std = runtime.predict_parameters(x, "UNC", seed)
    params = sample_parameters(mean, std, draws, np.random.default_rng(random_seed))
    expected = runtime.derive(params.reshape(-1, 4), np.tile(x, (draws, 1)))
    actual = runtime.retrieve(x, model_seed=seed, mc_samples=draws, random_seed=random_seed)
    for name in OUTPUT_ORDER:
        values = expected[name].reshape(draws, len(x))
        np.testing.assert_allclose(actual["uncertainty"][name]["std"], np.std(values, axis=0), atol=1e-12, rtol=1e-12)
        for key, q in (("q16", .16), ("q84", .84), ("q025", .025), ("q975", .975)):
            np.testing.assert_allclose(actual["uncertainty"][name][key], np.quantile(values, q, axis=0), atol=1e-12, rtol=1e-12)
    repeat = runtime.retrieve(x, model_seed=seed, mc_samples=draws, random_seed=random_seed)
    np.testing.assert_array_equal(actual["uncertainty"]["N"]["q16"], repeat["uncertainty"]["N"]["q16"])


def test_full_fast_preserves_group_rng(runtime, observations):
    frame, x = observations
    # Full bank and all 500 trees: neither LUT size nor tree count is reduced.
    actual = runtime.retrieve(x, "FAST", group_ids=frame["group_id_direct"].to_numpy())
    assert runtime._helper._load_fast().n_lut == 6291558
    assert runtime._helper._load_fast().n_rf_trees == 500
    expected = runtime._helper.retrieve(x, "FAST")  # These two rows belong to one profile.
    for name in OUTPUT_ORDER:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["uncertainty"] == {}


def test_fast_restarts_per_profile_and_preserves_order(runtime):
    frame = pd.read_csv(runtime.root / "examples/nasa/paper_input_113.csv").head(4)
    x = frame[list(INPUT_ORDER)].to_numpy(dtype=float)
    groups = frame["group_id_direct"].to_numpy()
    assert len(set(groups)) == 3
    actual = runtime.retrieve(x, "FAST", group_ids=groups)
    for group in dict.fromkeys(groups):
        keep = np.flatnonzero(groups == group)
        expected = runtime._helper.retrieve(x[keep], "FAST")
        for name in OUTPUT_ORDER:
            np.testing.assert_array_equal(actual[name][keep], expected[name])

