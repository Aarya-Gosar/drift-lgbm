import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from drift_lgbm.feature_importance import ClusteredImportanceReport
from drift_lgbm.model import DriftLGBMWarning, IncrementalLGBM

N_FEATURES = 8
FAST = dict(learning_rate=0.1, num_leaves=15, random_state=0, verbose=0)


def make_data(n, seed, shift=0.0, shifted=range(6)):
    """Nonlinear regression target; `shift` moves the mean of the `shifted` features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    X[:, list(shifted)] += shift
    y = 2 * X[:, 0] + 2 * np.sin(1.5 * X[:, 1]) + X[:, 2] * X[:, 3] + 0.5 * X[:, 4] + rng.normal(0, 0.3, n)
    cols = [f"f{j}" for j in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def rmse(model, X, y):
    return float(np.sqrt(np.mean((model.predict(X) - np.asarray(y)) ** 2)))


# --------------------------------------------------------------------------- fit == LGBMRegressor


@pytest.mark.parametrize("as_frame", [True, False])
def test_fit_matches_vanilla_lgbm_regressor(as_frame):
    X, y = make_data(3000, 0)
    X_test, _ = make_data(1000, 1)
    if not as_frame:
        X, X_test = X.to_numpy(), X_test.to_numpy()
    params = dict(learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=42)
    ours = IncrementalLGBM(task="regression", n_estimators_per_batch=120, verbose=0, **params).fit(X, y)
    vanilla = lgb.LGBMRegressor(n_estimators=120, verbose=-1, **params).fit(X, y)
    np.testing.assert_array_equal(ours.predict(X_test), vanilla.predict(X_test))
    np.testing.assert_array_equal(ours.feature_importances_, vanilla.feature_importances_)
    assert ours.n_trees_ == 120


def test_fit_matches_vanilla_with_eval_set_and_early_stopping():
    X, y = make_data(3000, 2)
    X_val, y_val = make_data(800, 3)
    X_test, _ = make_data(500, 4)
    kw = dict(learning_rate=0.3, num_leaves=31, random_state=1)
    callbacks = [lgb.early_stopping(5, verbose=False)]
    ours = IncrementalLGBM(n_estimators_per_batch=500, verbose=0, **kw).fit(
        X, y, eval_set=(X_val, y_val), callbacks=callbacks
    )
    vanilla = lgb.LGBMRegressor(n_estimators=500, verbose=-1, **kw).fit(
        X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(5, verbose=False)]
    )
    assert vanilla.best_iteration_ < 500  # early stopping actually kicked in
    np.testing.assert_array_equal(ours.predict(X_test), vanilla.predict(X_test))
    assert ours.n_trees_ == vanilla.best_iteration_
    assert "valid_0" in ours.evals_result_


def test_n_estimators_alias_is_drop_in():
    X, y = make_data(1000, 5)
    ours = IncrementalLGBM(n_estimators=30, verbose=0, random_state=0).fit(X, y)
    vanilla = lgb.LGBMRegressor(n_estimators=30, random_state=0, verbose=-1).fit(X, y)
    assert ours.n_trees_ == 30
    np.testing.assert_array_equal(ours.predict(X), vanilla.predict(X))


def test_classification_fit_matches_vanilla():
    X, y_cont = make_data(2000, 6)
    y = np.where(y_cont > 0, "up", "down")
    ours = IncrementalLGBM(task="classification", n_estimators_per_batch=40, verbose=0, random_state=0).fit(X, y)
    vanilla = lgb.LGBMClassifier(n_estimators=40, random_state=0, verbose=-1).fit(X, y)
    np.testing.assert_array_equal(ours.predict_proba(X), vanilla.predict_proba(X))
    np.testing.assert_array_equal(ours.predict(X), vanilla.predict(X))
    assert list(ours.classes_) == ["down", "up"]
    assert not hasattr(IncrementalLGBM(), "predict_proba")  # regression has no predict_proba


# --------------------------------------------------------------------------- incremental behaviour


def test_incremental_training_improves_over_first_batch():
    X, y = make_data(10_000, 7)
    X_test, y_test = make_data(3000, 8)
    first_only = IncrementalLGBM(n_estimators_per_batch=50, **FAST).fit(X.iloc[:2000], y.iloc[:2000])
    baseline = rmse(first_only, X_test, y_test)

    model = IncrementalLGBM(n_estimators_per_batch=50, **FAST).fit(X.iloc[:2000], y.iloc[:2000])
    for start in range(2000, 10_000, 2000):
        model.partial_fit(X.iloc[start : start + 2000], y.iloc[start : start + 2000])
    assert rmse(model, X_test, y_test) < 0.85 * baseline
    assert len(model.history_) == 5
    assert [r["stage"] for r in model.history_] == ["fit"] + ["partial_fit"] * 4
    assert all(r["validation"] in ("prequential", "holdout") for r in model.history_[1:])
    assert all(r["prequential_loss"] > 0 for r in model.history_[1:])
    assert all(not r["drift_detected"] for r in model.history_[1:])  # stationary data


def test_drift_triggers_rebin_and_performance_recovers():
    X_pre, y_pre = make_data(6000, 9)
    X_post, y_post = make_data(6000, 10, shift=2.5)
    X_post_test, y_post_test = make_data(2000, 11, shift=2.5)

    model = IncrementalLGBM(n_estimators_per_batch=60, **FAST).fit(X_pre.iloc[:3000], y_pre.iloc[:3000])
    model.partial_fit(X_pre.iloc[3000:], y_pre.iloc[3000:])
    assert not model.history_[-1]["drift_detected"]
    before = rmse(model, X_post_test, y_post_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        model.partial_fit(X_post.iloc[:3000], y_post.iloc[:3000])
    shift_record = model.history_[-1]
    assert shift_record["drift_detected"] and shift_record["rebin_triggered"]
    assert shift_record["n_drifted_features"] == 6
    assert set(shift_record["drifted_features"]) == {f"f{j}" for j in range(6)}
    assert shift_record["action"] in ("rebin", "rebin_rejected")

    # The tracker was reset to the new regime, so the next post-shift batch is not "drift".
    model.partial_fit(X_post.iloc[3000:], y_post.iloc[3000:])
    assert not model.history_[-1]["drift_detected"]
    after = rmse(model, X_post_test, y_post_test)
    assert after < 0.5 * before


def test_rebin_after_abrupt_shift_excludes_dead_regime_data():
    X_pre, y_pre = make_data(4000, 12)
    X_post, y_post = make_data(2000, 13, shift=3.0)
    model = IncrementalLGBM(n_estimators_per_batch=40, safety_checks=False, **FAST)
    model.fit(X_pre.iloc[:2000], y_pre.iloc[:2000])
    model.partial_fit(X_pre.iloc[2000:], y_pre.iloc[2000:])
    trees_before = model.n_trees_
    model.partial_fit(X_post, y_post)
    record = model.history_[-1]
    assert record["action"] == "rebin" and record["rebin_accepted"] is True
    # Both stored batches come from the old regime, so the fresh trees see only the new batch.
    assert record["window_batches_used"] == 0 and record["train_rows_used"] == 2000
    # Old trees that still help are kept ("ported"); never more than 50% are removed.
    assert record["trees_after"] >= trees_before // 2
    assert record["trees_retired"] + record["trees_evicted"] <= trees_before // 2


def test_rebin_uses_stored_batches_from_the_current_regime():
    # Gradual drift: batch 1 shifts 2/8 features (25%, below the 30% rebin
    # threshold); batch 2 shifts one more (3/8 = 37.5% vs the reference -> rebin).
    # Batch 1 already looks like batch 2, so it joins the rebin window; batch 0 does not.
    model = IncrementalLGBM(n_estimators_per_batch=30, safety_checks=False, **FAST)
    model.fit(*make_data(2000, 120))
    model.partial_fit(*make_data(2000, 121, shift=3.0, shifted=[0, 1]))
    assert model.history_[-1]["action"] == "incremental"
    model.partial_fit(*make_data(2000, 122, shift=3.0, shifted=[0, 1, 2]))
    record = model.history_[-1]
    assert record["action"] == "rebin"
    assert record["window_batches_used"] == 1
    assert record["train_rows_used"] == 4000


def test_compatible_stored_batches_unit():
    model = IncrementalLGBM(n_estimators_per_batch=10, **FAST).fit(*make_data(1500, 123))
    model.partial_fit(*make_data(1500, 124))
    same_regime = make_data(1500, 125)[0].to_numpy()
    new_regime = make_data(1500, 126, shift=3.0)[0].to_numpy()
    assert len(model._compatible_stored_batches(same_regime)) == 2
    assert len(model._compatible_stored_batches(new_regime)) == 0
    X_new, y_new = np.zeros((5, N_FEATURES)), np.ones(5)
    X_all, y_all, w_all = model._rebin_training_data(X_new, y_new, None, stored=list(model._stored_batches))
    assert X_all.shape == (1500 + 1500 + 5, N_FEATURES) and w_all is None
    assert model._rebin_training_data(X_new, y_new, None, stored=[])[0] is X_new


def test_safety_check_rejects_harmful_rebin(monkeypatch):
    X_pre, y_pre = make_data(3000, 14)
    X_post, y_post = make_data(3000, 15, shift=3.0)
    model = IncrementalLGBM(n_estimators_per_batch=40, **FAST).fit(X_pre, y_pre)

    def poisoned_window(self, X_tr, y_tr, w_tr, stored=None):  # rebin candidate trains on garbage labels
        return X_tr, np.random.default_rng(0).permutation(y_tr) * 10, w_tr

    monkeypatch.setattr(IncrementalLGBM, "_rebin_training_data", poisoned_window)
    with pytest.warns(DriftLGBMWarning, match="rebin made holdout rmse worse"):
        model.partial_fit(X_post, y_post)
    record = model.history_[-1]
    assert record["action"] == "rebin_rejected"
    assert record["rebin_accepted"] is False
    assert record["validation"] == "holdout" and record["n_val_rows"] == 600
    assert record["candidate_loss_ours"] > record["candidate_loss_naive"]
    # The naive strategy was refit on the whole batch.
    assert record["train_rows_used"] == 3000
    assert record["notes"]

    # Same check with an explicit eval_set.
    model2 = IncrementalLGBM(n_estimators_per_batch=40, **FAST).fit(X_pre, y_pre)
    X_val, y_val = make_data(800, 16, shift=3.0)
    with pytest.warns(DriftLGBMWarning, match="rebin made eval_set rmse worse"):
        model2.partial_fit(X_post, y_post, eval_set=(X_val, y_val))
    record = model2.history_[-1]
    assert record["action"] == "rebin_rejected"
    assert record["val_loss_after"] == pytest.approx(record["candidate_loss_naive"])


def test_poison_batch_trees_are_retired():
    X, y = make_data(3000, 16)
    X_val, y_val = make_data(1500, 17)
    model = IncrementalLGBM(n_estimators_per_batch=100, **FAST).fit(X, y)
    X_noise, _ = make_data(3000, 18)
    y_noise = pd.Series(np.random.default_rng(19).normal(0, 6, 3000))

    naive = lgb.train(model._train_params, lgb.Dataset(X_noise, y_noise), 100, init_model=model.booster_)
    naive_rmse = float(np.sqrt(np.mean((naive.predict(X_val) - y_val) ** 2)))

    model.partial_fit(X_noise, y_noise, eval_set=(X_val, y_val))
    record = model.history_[-1]
    assert record["action"] == "incremental"
    # The 50%-per-call guardrail caps retirement at 50 of the 100 existing trees...
    assert record["trees_retired"] == 50
    assert rmse(model, X_val, y_val) < 0.5 * naive_rmse
    # ...and the next (clean) batch retires the remaining poison trees before training.
    model.partial_fit(*make_data(3000, 22), eval_set=(X_val, y_val))
    assert model.history_[-1]["trees_retired"] >= 40
    assert rmse(model, X_val, y_val) < 1.1 * record["val_loss_before"]


def test_max_total_trees_never_exceeded():
    model = IncrementalLGBM(n_estimators_per_batch=100, max_total_trees=250, **FAST)
    X, y = make_data(1500, 20)
    model.fit(X, y)
    for i in range(10):
        Xb, yb = make_data(1500, 100 + i)
        model.partial_fit(Xb, yb)
        assert model.n_trees_ <= 250
    assert all(r["trees_after"] <= 250 for r in model.history_)
    assert sum(r["trees_evicted"] for r in model.history_) > 0
    # A cap equal to the batch size still works (fewer new trees when the 50% guardrail binds).
    tight = IncrementalLGBM(n_estimators_per_batch=50, max_total_trees=50, **FAST).fit(X, y)
    for i in range(3):
        tight.partial_fit(*make_data(1000, 200 + i))
        assert tight.n_trees_ <= 50


def test_feature_importance_report_after_incremental_training():
    X, y = make_data(2000, 21)
    model = IncrementalLGBM(n_estimators_per_batch=40, **FAST).fit(X, y)
    for i in range(3):
        model.partial_fit(*make_data(2000, 30 + i))
    X_val, y_val = make_data(1000, 40)
    report = model.feature_importance_report(X_val, y_val)
    assert isinstance(report, ClusteredImportanceReport)
    assert set(report.feature_importance) == set(X.columns)
    assert report.stability_scores is not None and report.n_snapshots == 4
    assert set(report.stability_scores) == set(report.clusters)
    assert report.cluster_permutation_importance is not None
    assert report.total_importance == pytest.approx(model.booster_.feature_importance("gain").sum())
    assert isinstance(report.correlation_matrix, pd.DataFrame)
    assert len(model.importance_snapshots_) == 4
    # f0 (linear, weight 2) is among the most important clusters.
    top = report.to_frame().index[:3]
    assert report.cluster_of("f0") in top
    report.to_string()


def test_serialization_round_trip():
    X, y = make_data(2000, 50)
    model = IncrementalLGBM(n_estimators_per_batch=30, **FAST).fit(X, y)
    model.partial_fit(*make_data(2000, 51))
    X_test, _ = make_data(500, 52)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X_test), model.predict(X_test))
    assert restored.history_ == model.history_
    # The full training state survives: the same next update gives the same model.
    nxt = make_data(2000, 53, shift=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftLGBMWarning)
        model.partial_fit(*nxt)
        restored.partial_fit(*nxt)
    np.testing.assert_array_equal(restored.predict(X_test), model.predict(X_test))


# --------------------------------------------------------------------------- API details


def test_validation_modes():
    X, y = make_data(2000, 60)
    model = IncrementalLGBM(n_estimators_per_batch=20, **FAST).fit(X, y)
    model.partial_fit(*make_data(1000, 61))
    record = model.history_[-1]
    assert record["validation"] in ("prequential", "holdout")
    assert record["train_rows_used"] == 1000  # never loses training rows without an eval_set

    Xv, yv = make_data(300, 62)
    model.partial_fit(*make_data(1000, 63), eval_set=[(Xv, yv)])
    record = model.history_[-1]
    assert record["validation"] == "eval_set" and record["n_val_rows"] == 300
    assert record["val_loss_before"] is not None and record["val_loss_after"] is not None

    # A shift forces a decision: candidates are compared on the most recent 20%.
    model.partial_fit(*make_data(1000, 64, shift=3.0))
    record = model.history_[-1]
    assert record["validation"] == "holdout" and record["n_val_rows"] == 200
    assert record["candidate_loss_ours"] is not None and record["candidate_loss_naive"] is not None
    assert record["train_rows_used"] >= 1000

    model.set_params(validation_fraction=0.0)
    model.partial_fit(*make_data(1000, 65, shift=-3.0))
    record = model.history_[-1]
    assert record["validation"] == "prequential" and record["candidate_loss_naive"] is None
    assert any("without the naive-baseline safety check" in n for n in record["notes"])


def test_stored_batches_are_bounded():
    X, y = make_data(3000, 70)
    model = IncrementalLGBM(n_estimators_per_batch=10, max_stored_batches=2, max_rows_per_stored_batch=500, **FAST)
    model.fit(X, y)
    for i in range(4):
        model.partial_fit(*make_data(1200, 71 + i))
    assert len(model._stored_batches) == 2
    assert all(b[0].shape == (500, N_FEATURES) and isinstance(b[0], np.ndarray) for b in model._stored_batches)
    no_store = IncrementalLGBM(n_estimators_per_batch=10, store_recent_data=False, **FAST).fit(X, y)
    assert no_store._stored_batches is None


def test_partial_fit_before_fit_equals_fit():
    X, y = make_data(1500, 80)
    a = IncrementalLGBM(n_estimators_per_batch=25, **FAST).partial_fit(X, y)
    b = IncrementalLGBM(n_estimators_per_batch=25, **FAST).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert a.history_[0]["stage"] == "fit"


def test_sklearn_params_clone_and_repr():
    model = IncrementalLGBM(task="regression", n_estimators_per_batch=10, num_leaves=7, learning_rate=0.2, verbose=0)
    params = model.get_params()
    assert params["num_leaves"] == 7 and params["n_estimators_per_batch"] == 10 and params["drift_fraction"] == 0.3
    model.set_params(num_leaves=9, drift_fraction=0.4)
    assert model.get_params()["num_leaves"] == 9 and model.drift_fraction == 0.4
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not cloned.__sklearn_is_fitted__()
    assert "num_leaves=9" in repr(model)
    X, y = make_data(800, 81)
    model.fit(X, y)
    assert 0.5 < model.score(X, y) <= 1.0


def test_classification_incremental():
    X, y_cont = make_data(3000, 82)
    y = np.where(y_cont > 0, "up", "down")
    model = IncrementalLGBM(task="classification", n_estimators_per_batch=30, **FAST).fit(X, y)
    X2, y2_cont = make_data(3000, 83)
    model.partial_fit(X2, np.where(y2_cont > 0, "up", "down"))
    assert model.history_[-1]["metric"] == "logloss"
    X_test, y_test_cont = make_data(1000, 84)
    y_test = np.where(y_test_cont > 0, "up", "down")
    assert set(model.predict(X_test)) <= {"up", "down"}
    proba = model.predict_proba(X_test)
    assert proba.shape == (1000, 2) and np.allclose(proba.sum(axis=1), 1.0)
    assert model.score(X_test, y_test) > 0.8
    with pytest.raises(ValueError, match="not seen"):
        model.partial_fit(X2, np.where(y2_cont > 0, "up", "sideways"))
    auc_model = IncrementalLGBM(task="classification", n_estimators_per_batch=20, scoring_metric="auc", **FAST)
    auc_model.fit(X, y).partial_fit(X2, np.where(y2_cont > 0, "up", "down"))
    assert auc_model.history_[-1]["metric"] == "auc"


def test_column_alignment_and_validation():
    X, y = make_data(1000, 90)
    model = IncrementalLGBM(n_estimators_per_batch=20, **FAST).fit(X, y)
    np.testing.assert_array_equal(model.predict(X[X.columns[::-1]]), model.predict(X))
    with pytest.raises(ValueError):
        model.predict(X.drop(columns=["f0"]))
    with pytest.raises(RuntimeError):
        IncrementalLGBM().predict(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(task="ranking"),
        dict(n_estimators_per_batch=200, max_total_trees=100),
        dict(drift_threshold=0.0),
        dict(num_boost_round=10),
        dict(task="classification", objective="regression"),
        dict(validation_fraction=1.0),
    ],
)
def test_invalid_parameters(kwargs):
    X, y = make_data(200, 91)
    with pytest.raises(ValueError):
        IncrementalLGBM(verbose=0, **kwargs).fit(X, y)


def test_multiclass_rejected():
    X, _ = make_data(300, 92)
    with pytest.raises(NotImplementedError):
        IncrementalLGBM(task="classification", verbose=0).fit(X, np.arange(300) % 3)


def test_verbose_logging(capsys):
    X, y = make_data(1000, 93)
    model = IncrementalLGBM(n_estimators_per_batch=10, learning_rate=0.1, verbose=1).fit(X, y)
    model.partial_fit(*make_data(1000, 94))
    out = capsys.readouterr().out
    assert "[drift-lgbm] fit: 10 trees" in out
    assert "[drift-lgbm] batch 1" in out and "incremental" in out
