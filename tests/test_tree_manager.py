import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from drift_lgbm.tree_manager import TreeManager, per_tree_importance

N_CLEAN = 60
N_POISON = 10
PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=15, verbose=-1, seed=0)


def _clean_data(rng, n):
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, 0.1, n)
    return X, y


def _rmse(booster, X, y):
    return float(np.sqrt(np.mean((booster.predict(X) - y) ** 2)))


@pytest.fixture(scope="module")
def poisoned():
    """A clean 60-tree model plus 10 'poison' trees trained via init_model on pure noise."""
    rng = np.random.default_rng(0)
    X_tr, y_tr = _clean_data(rng, 3000)
    X_val, y_val = _clean_data(rng, 2000)
    clean = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), N_CLEAN)
    X_noise, y_noise = rng.normal(size=(1000, 5)), rng.normal(0, 5, 1000)
    booster = lgb.train(PARAMS, lgb.Dataset(X_noise, y_noise), N_POISON, init_model=clean)
    assert booster.num_trees() == N_CLEAN + N_POISON
    return dict(clean=clean, booster=booster, X_val=X_val, y_val=y_val)


def test_score_trees_ranks_poison_trees_at_bottom(poisoned):
    tm = TreeManager(poisoned["booster"])
    ranking = tm.score_trees(poisoned["X_val"], poisoned["y_val"], metric="rmse")
    assert len(ranking) == N_CLEAN + N_POISON
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)  # most helpful first
    bottom = {idx for idx, _ in ranking[-N_POISON:]}
    assert bottom == set(range(N_CLEAN, N_CLEAN + N_POISON))
    assert all(s < 0 for _, s in ranking[-N_POISON:])
    assert tm.metric_ == "rmse"
    assert tm.baseline_ == pytest.approx(_rmse(poisoned["booster"], poisoned["X_val"], poisoned["y_val"]))


def test_contribution_is_exact_leave_one_out(poisoned):
    booster, X_val, y_val = poisoned["booster"], poisoned["X_val"], poisoned["y_val"]
    tm = TreeManager(booster)
    tm.score_trees(X_val, y_val, metric="rmse")
    full = booster.predict(X_val, raw_score=True)
    for j in [0, 7, 42, 65]:
        tree_j = booster.predict(X_val, raw_score=True, start_iteration=j, num_iteration=1)
        loo = np.sqrt(np.mean((full - tree_j - y_val) ** 2))
        assert tm.scores_[j] == pytest.approx(loo - tm.baseline_, rel=1e-9, abs=1e-12)


def test_retire_trees_produces_valid_booster(poisoned):
    booster, X_val = poisoned["booster"], poisoned["X_val"]
    tm = TreeManager(booster)
    new = tm.retire_trees(booster, list(range(N_CLEAN, N_CLEAN + N_POISON)))
    assert isinstance(new, lgb.Booster)
    assert new.num_trees() == N_CLEAN
    # Removing exactly the poison trees recovers the clean model's predictions.
    np.testing.assert_allclose(new.predict(X_val), poisoned["clean"].predict(X_val), rtol=0, atol=1e-10)
    # The new booster serialises and can continue training.
    restored = pickle.loads(pickle.dumps(new))
    np.testing.assert_allclose(restored.predict(X_val), new.predict(X_val), rtol=0, atol=0)
    rng = np.random.default_rng(1)
    X_more, y_more = _clean_data(rng, 500)
    continued = lgb.train(PARAMS, lgb.Dataset(X_more, y_more), 5, init_model=new)
    assert continued.num_trees() == N_CLEAN + 5


def test_retire_trees_removes_middle_trees_exactly(poisoned):
    booster, X_val = poisoned["booster"], poisoned["X_val"]
    removed = [3, 17, 18, 40]
    new = TreeManager().retire_trees(booster, removed)
    expected = booster.predict(X_val, raw_score=True)
    for j in removed:
        expected -= booster.predict(X_val, raw_score=True, start_iteration=j, num_iteration=1)
    np.testing.assert_allclose(new.predict(X_val, raw_score=True), expected, rtol=0, atol=1e-10)


def test_retiring_negative_trees_improves_validation(poisoned):
    booster, X_val, y_val = poisoned["booster"], poisoned["X_val"], poisoned["y_val"]
    tm = TreeManager(booster)
    tm.score_trees(X_val, y_val)
    retiring = tm.identify_retiring(threshold=0.0)
    assert set(range(N_CLEAN, N_CLEAN + N_POISON)) <= set(retiring)
    new = tm.retire_trees(booster, retiring)
    assert _rmse(new, X_val, y_val) <= _rmse(booster, X_val, y_val)

    greedy = tm.select_retirement(threshold=0.0, greedy=True)
    new_greedy = tm.retire_trees(booster, greedy)
    assert _rmse(new_greedy, X_val, y_val) < _rmse(booster, X_val, y_val)


def test_identify_retiring_threshold(poisoned):
    tm = TreeManager(poisoned["booster"])
    tm.score_trees(poisoned["X_val"], poisoned["y_val"])
    worst_first = tm.identify_retiring(threshold=0.0)
    assert [tm.scores_[i] for i in worst_first] == sorted(tm.scores_[i] for i in worst_first)
    assert len(tm.identify_retiring(threshold=-np.inf)) == 0
    assert len(tm.identify_retiring(threshold=np.inf)) == N_CLEAN + N_POISON


def test_fifty_percent_guardrail(poisoned):
    booster = poisoned["booster"]
    tm = TreeManager(booster)
    tm.score_trees(poisoned["X_val"], poisoned["y_val"])
    everything = tm.identify_retiring(threshold=np.inf)  # all 70 trees, worst first
    with pytest.warns(UserWarning, match="guardrail"):
        new = tm.retire_trees(booster, everything)
    assert new.num_trees() == (N_CLEAN + N_POISON) // 2
    # The worst trees (the poison ones) are among those retired.
    kept_expected = sorted(set(range(N_CLEAN + N_POISON)) - set(everything[:35]))
    X_val = poisoned["X_val"]
    expected = sum(
        booster.predict(X_val, raw_score=True, start_iteration=j, num_iteration=1) for j in kept_expected
    )
    np.testing.assert_allclose(new.predict(X_val, raw_score=True), expected, rtol=0, atol=1e-9)
    # select_retirement respects the same cap.
    assert len(tm.select_retirement(threshold=np.inf, greedy=False)) == 35
    # Tiny ensembles: a single tree can never be retired.
    single = lgb.train(PARAMS, lgb.Dataset(X_val, poisoned["y_val"]), 1)
    with pytest.warns(UserWarning):
        assert TreeManager().retire_trees(single, [0]).num_trees() == 1


def test_retire_trees_never_mutates_input(poisoned):
    booster, X_val = poisoned["booster"], poisoned["X_val"]
    before_text = booster.model_to_string()
    before_pred = booster.predict(X_val)
    new = TreeManager().retire_trees(booster, [0, 1, 2])
    assert new is not booster
    assert booster.num_trees() == N_CLEAN + N_POISON
    assert booster.model_to_string() == before_text
    np.testing.assert_array_equal(booster.predict(X_val), before_pred)
    assert TreeManager().retire_trees(booster, []) is not booster


def test_retire_trees_validates_indices(poisoned):
    tm = TreeManager(poisoned["booster"])
    with pytest.raises(ValueError):
        tm.retire_trees(None, [1000])
    with pytest.raises(ValueError):
        tm.retire_trees(None, [1, 1])
    with pytest.raises(RuntimeError):
        TreeManager().identify_retiring()


def test_select_retirement_greedy_never_worse(poisoned):
    booster, X_val, y_val = poisoned["booster"], poisoned["X_val"], poisoned["y_val"]
    tm = TreeManager(booster)
    tm.score_trees(X_val, y_val, metric="mae")
    selected = tm.select_retirement(threshold=np.inf, greedy=True)  # every tree is a candidate
    new = tm.retire_trees(booster, selected)
    mae = lambda b: float(np.mean(np.abs(b.predict(X_val) - y_val)))  # noqa: E731
    assert mae(new) <= mae(booster)
    assert set(range(N_CLEAN, N_CLEAN + N_POISON)) <= set(selected)


def test_mae_and_callable_metrics(poisoned):
    booster, X_val, y_val = poisoned["booster"], poisoned["X_val"], poisoned["y_val"]
    tm = TreeManager(booster)
    mae_rank = tm.score_trees(X_val, y_val, metric="mae")
    assert {i for i, _ in mae_rank[-N_POISON:]} == set(range(N_CLEAN, N_CLEAN + N_POISON))

    rmse_order = [i for i, _ in tm.score_trees(X_val, y_val, metric="rmse")]
    mse = lambda y, p: float(np.mean((y - p) ** 2))  # noqa: E731
    mse_order = [i for i, _ in tm.score_trees(X_val, y_val, metric=mse)]
    assert mse_order == rmse_order  # MSE is a monotone transform of RMSE
    with pytest.raises(ValueError):
        tm.score_trees(X_val, y_val, metric="nonsense")


def test_binary_classification_metrics():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4000, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 4000) > 0).astype(int)
    X_val = rng.normal(size=(2000, 4))
    y_val = (X_val[:, 0] + 0.5 * X_val[:, 1] + rng.normal(0, 0.5, 2000) > 0).astype(int)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=15, verbose=-1, seed=0)
    clean = lgb.train(params, lgb.Dataset(X, y), 40)
    # Poison: 8 trees fitted to *inverted* labels.
    poison_X = rng.normal(size=(800, 4))
    booster = lgb.train(params, lgb.Dataset(poison_X, (poison_X[:, 0] <= 0).astype(int)), 8, init_model=clean)

    tm = TreeManager(booster)
    ranking = tm.score_trees(X_val, y_val)  # default metric for binary objective
    assert tm.metric_ == "logloss"
    assert {i for i, _ in ranking[-8:]} == set(range(40, 48))

    auc_ranking = tm.score_trees(X_val, y_val, metric="auc")
    auc_scores = dict(auc_ranking)
    assert np.mean([auc_scores[i] for i in range(40, 48)]) < np.mean([auc_scores[i] for i in range(40)])
    # The inverted poison trees drag the ensemble's AUC below 0.5; retiring them restores it.
    assert tm.baseline_ < 0.5
    repaired = tm.retire_trees(booster, list(range(40, 48)))
    tm_repaired = TreeManager(repaired)
    tm_repaired.score_trees(X_val, y_val, metric="auc")
    assert tm_repaired.baseline_ > 0.9
    with pytest.raises(ValueError):
        tm.score_trees(X_val, np.where(y_val == 1, 5, 0), metric="logloss")


def test_linear_tree_fallback():
    rng = np.random.default_rng(3)
    X, y = _clean_data(rng, 1500)
    params = dict(PARAMS, linear_tree=True)
    booster = lgb.train(params, lgb.Dataset(X, y), 15)
    tm = TreeManager(booster)
    tm.score_trees(X, y)
    full = booster.predict(X, raw_score=True)
    tree_3 = booster.predict(X, raw_score=True, start_iteration=3, num_iteration=1)
    expected = np.sqrt(np.mean((full - tree_3 - y) ** 2)) - np.sqrt(np.mean((full - y) ** 2))
    assert tm.scores_[3] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    new = tm.retire_trees(booster, [3])
    np.testing.assert_allclose(new.predict(X, raw_score=True), full - tree_3, rtol=0, atol=1e-9)


def test_dataframe_features_are_aligned_by_name():
    rng = np.random.default_rng(4)
    X, y = _clean_data(rng, 1500)
    frame = pd.DataFrame(X, columns=list("abcde"))
    booster = lgb.train(PARAMS, lgb.Dataset(frame, y), 20)
    tm = TreeManager(booster)
    by_array = dict(tm.score_trees(X, y))
    by_shuffled_frame = dict(tm.score_trees(frame[list("edcba")], y))
    assert by_array == pytest.approx(by_shuffled_frame)


def test_accepts_sklearn_estimator():
    rng = np.random.default_rng(5)
    X, y = _clean_data(rng, 1000)
    est = lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(X, y)
    assert len(TreeManager(est).score_trees(X, y)) == 10


def test_per_tree_importance_matches_booster(poisoned):
    booster = poisoned["booster"]
    gains = per_tree_importance(booster, "gain")
    splits = per_tree_importance(booster, "split")
    assert gains.shape == (N_CLEAN + N_POISON, 5)
    np.testing.assert_allclose(gains.sum(axis=0), booster.feature_importance("gain"), rtol=1e-4)
    np.testing.assert_array_equal(splits.sum(axis=0), booster.feature_importance("split"))


def test_multiclass_not_supported():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(600, 3))
    y = rng.integers(0, 3, 600)
    booster = lgb.train(dict(objective="multiclass", num_class=3, verbose=-1), lgb.Dataset(X, y), 3)
    with pytest.raises(NotImplementedError):
        TreeManager(booster).score_trees(X, y)


def test_constant_model_single_leaf_trees():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 3))
    booster = lgb.train(PARAMS, lgb.Dataset(X, np.full(500, 2.0)), 5)
    tm = TreeManager(booster)
    ranking = tm.score_trees(X, np.full(500, 2.0))
    assert len(ranking) == booster.num_trees()
    assert per_tree_importance(booster).sum() == 0.0


def test_select_retirement_significance_gate(poisoned):
    booster, X_val, y_val = poisoned["booster"], poisoned["X_val"], poisoned["y_val"]
    tm = TreeManager(booster)
    tm.score_trees(X_val, y_val)
    # Strongly harmful trees survive the significance gate.
    gated = tm.select_retirement(threshold=0.0, min_z=2.0)
    assert set(range(N_CLEAN, N_CLEAN + N_POISON)) <= set(gated)

    # On an over-fitted clean model, noise alone makes some trees look harmful;
    # the gate retires far fewer of them.
    rng = np.random.default_rng(11)
    X_tr, y_tr = _clean_data(rng, 600)
    X_v, y_v = _clean_data(rng, 600)
    overfit = lgb.train(dict(PARAMS, num_leaves=63, min_data_in_leaf=2), lgb.Dataset(X_tr, y_tr), 300)
    tm = TreeManager(overfit)
    tm.score_trees(X_v, y_v)
    loose = tm.select_retirement(threshold=0.0, min_z=0.0)
    strict = tm.select_retirement(threshold=0.0, min_z=2.0)
    assert len(strict) < len(loose)
    rmse = lambda b: float(np.sqrt(np.mean((b.predict(X_v) - y_v) ** 2)))  # noqa: E731
    assert rmse(tm.retire_trees(overfit, strict)) <= rmse(overfit)
