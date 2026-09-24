"""End-to-end tests on synthetic financial data (drift_lgbm.testing)."""

import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pytest

from drift_lgbm.model import DriftLGBMWarning, IncrementalLGBM
from drift_lgbm.testing import generate_financial_data

LGBM = dict(learning_rate=0.1, num_leaves=31, random_state=0)


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def _batches(X, y, n_batches, batch_size):
    for b in range(n_batches):
        rows = slice(b * batch_size, (b + 1) * batch_size)
        yield X.iloc[rows], y.iloc[rows]


def test_full_pipeline_runs_end_to_end():
    X, y, meta = generate_financial_data(n_samples=14_000, regime_change_at=8_000, random_state=0)
    X_val, y_val = X.iloc[12_000:13_000], y.iloc[12_000:13_000]
    X_test, y_test = X.iloc[13_000:], y.iloc[13_000:]
    model = IncrementalLGBM(n_estimators_per_batch=50, verbose=0, **LGBM)
    model.fit(X.iloc[:2000], y.iloc[:2000], eval_set=(X_val, y_val))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        for Xb, yb in _batches(X.iloc[2000:12_000], y.iloc[2000:12_000], 5, 2000):
            model.partial_fit(Xb, yb)
    preds = model.predict(X_test)
    assert preds.shape == (1000,) and np.all(np.isfinite(preds))
    report = model.feature_importance_report(X_val, y_val)
    assert report.n_snapshots == 6 and report.stability_scores is not None
    assert len(model.history_) == 6 and len(model.drift_reports_) == 6
    assert any(r["drift_detected"] for r in model.history_)  # the regime change was seen
    assert model.n_trees_ <= model.max_total_trees
    # Everything survives a pickle round trip.
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X_test), preds)
    report.to_string()


def test_regime_change_recovery():
    X, y, _ = generate_financial_data(n_samples=20_000, regime_change_at=8_000, random_state=1)
    X_post_test, y_post_test = X.iloc[16_000:], y.iloc[16_000:]
    model = IncrementalLGBM(n_estimators_per_batch=100, verbose=0, **LGBM).fit(X.iloc[:8000], y.iloc[:8000])
    pre_change_rmse = _rmse(model.predict(X_post_test), y_post_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        model.partial_fit(X.iloc[8000:12_000], y.iloc[8000:12_000])
        assert model.history_[-1]["rebin_triggered"]
        model.partial_fit(X.iloc[12_000:16_000], y.iloc[12_000:16_000])
    adapted_rmse = _rmse(model.predict(X_post_test), y_post_test)
    assert adapted_rmse < 0.5 * pre_change_rmse


def test_clustered_importance_identifies_informative_groups():
    X, y, meta = generate_financial_data(
        n_samples=12_000,
        n_correlated_groups=4,
        n_informative_groups=2,
        features_per_group=5,
        n_informative=4,
        n_noise=6,
        random_state=2,
    )
    model = IncrementalLGBM(n_estimators_per_batch=100, verbose=0, **LGBM).fit(X.iloc[:5000], y.iloc[:5000])
    model.partial_fit(X.iloc[5000:10_000], y.iloc[5000:10_000])
    X_val, y_val = X.iloc[10_000:], y.iloc[10_000:]
    report = model.feature_importance_report(X_val, y_val)

    # Every correlated family is recovered as exactly one cluster.
    for family, features in meta["groups"].items():
        cluster = report.cluster_of(features[0])
        assert sorted(report.clusters[cluster]) == sorted(features), family
    share = report.cluster_share
    perm = report.cluster_permutation_importance
    informative = [report.cluster_of(meta["groups"][g][0]) for g in meta["informative_groups"]]
    uninformative = [report.cluster_of(meta["groups"][g][0]) for g in meta["uninformative_groups"]]
    noise = [report.cluster_of(f) for f in meta["noise_features"]]
    assert min(share[c] for c in informative) > 0.05
    assert max(share[c] for c in uninformative + noise) < 0.01
    assert min(perm[c] for c in informative) > 10 * max(abs(perm[c]) for c in uninformative + noise)
    # Both informative factors carry the same true weight (|coef| ~ 1.38 for this
    # seed), and the cluster view splits credit evenly even though the
    # individual features within each cluster get very different shares.
    coefs = [abs(meta["coefficients"][g]) for g in meta["informative_groups"]]
    assert abs(coefs[0] - coefs[1]) < 0.01
    assert abs(share[informative[0]] - share[informative[1]]) < 0.05


def _naive_vs_ours(concept_shift, seed):
    n_batches, batch = 5, 5000
    X, y, _ = generate_financial_data(
        n_samples=n_batches * batch + 5000,
        n_correlated_groups=4,
        features_per_group=5,
        n_informative=8,
        n_noise=12,
        regime_change_at=(n_batches * batch) // 2,
        regime_concept_shift=concept_shift,
        random_state=seed,
    )
    n_train = n_batches * batch
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]  # latest (post-change) distribution

    params = dict(objective="regression", learning_rate=0.1, num_leaves=31, seed=0, verbosity=-1)
    naive = None
    for Xb, yb in _batches(X, y, n_batches, batch):
        naive = lgb.train(params, lgb.Dataset(Xb, yb), 100, init_model=naive)

    ours = IncrementalLGBM(n_estimators_per_batch=100, verbose=0, **LGBM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        for b, (Xb, yb) in enumerate(_batches(X, y, n_batches, batch)):
            (ours.fit if b == 0 else ours.partial_fit)(Xb, yb)
    return _rmse(naive.predict(X_test), y_test), _rmse(ours.predict(X_test), y_test), ours


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_naive_init_model_under_covariate_shift(seed):
    naive_rmse, ours_rmse, ours = _naive_vs_ours(concept_shift=0.0, seed=seed)
    assert any(r["rebin_triggered"] for r in ours.history_)
    # y|x is unchanged, so old trees stay valid: we must at least match naive init_model.
    assert ours_rmse <= 1.005 * naive_rmse


@pytest.mark.parametrize("seed", [0, 1])
def test_beats_naive_init_model_under_concept_drift(seed):
    naive_rmse, ours_rmse, ours = _naive_vs_ours(concept_shift=0.5, seed=seed)
    # Half the factor premia flip: dead-regime trees actively hurt and get retired.
    assert sum(r["trees_retired"] for r in ours.history_) > 20
    assert ours_rmse < 0.97 * naive_rmse


def test_classification_pipeline():
    X, y_cont, _ = generate_financial_data(n_samples=12_000, regime_change_at=6_000, random_state=3)
    y = np.where(y_cont > y_cont.median(), "long", "short")
    model = IncrementalLGBM(task="classification", n_estimators_per_batch=50, verbose=0, **LGBM)
    model.fit(X.iloc[:4000], y[:4000])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        model.partial_fit(X.iloc[4000:8000], y[4000:8000])
        model.partial_fit(X.iloc[8000:10_000], y[8000:10_000], eval_set=(X.iloc[10_000:11_000], y[10_000:11_000]))
    X_test, y_test = X.iloc[11_000:], y[11_000:]
    assert model.score(X_test, y_test) > 0.75
    assert model.predict_proba(X_test).shape == (1000, 2)
    report = model.feature_importance_report(X_test, y_test)
    assert report.permutation_metric == "logloss"
