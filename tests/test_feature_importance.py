import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from drift_lgbm.feature_importance import ClusteredImportance, ClusteredImportanceReport

GROUPS = "ABC"
LGBM_KW = dict(n_estimators=80, learning_rate=0.1, num_leaves=15, random_state=0, verbose=-1)


def _grouped_data(n=3000, noise_levels=(0.2, 0.35, 0.5, 0.65, 0.8), n_noise=3, seed=0):
    """3 latent signals; each observed through 5 noisy copies; plus pure-noise features."""
    rng = np.random.default_rng(seed)
    signals = rng.normal(size=(n, len(GROUPS)))
    eps = rng.normal(size=(n, len(GROUPS), len(noise_levels)))
    cols = {}
    for g, name in enumerate(GROUPS):
        for k, sigma in enumerate(noise_levels):
            cols[f"{name}{k}"] = signals[:, g] + sigma * eps[:, g, k]
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = 3 * signals[:, 0] + 2 * signals[:, 1] + 1 * signals[:, 2] + 0.1 * rng.normal(size=n)
    return X, pd.Series(y, name="y")


@pytest.fixture(scope="module")
def fitted():
    X, y = _grouped_data()
    model = lgb.LGBMRegressor(**LGBM_KW).fit(X, y)
    return model, X, y


def _as_sets(clusters):
    return {frozenset(members) for members in clusters.values()}


def test_clustering_recovers_known_groups(fitted):
    model, X, _ = fitted
    report = ClusteredImportance().compute(model, X)
    expected = {frozenset(f"{g}{k}" for k in range(5)) for g in GROUPS}
    expected |= {frozenset([f"noise{i}"]) for i in range(3)}
    assert _as_sets(report.clusters) == expected
    assert report.n_clusters == 6
    # Deterministic ids: numbered by first appearance in column order.
    assert report.clusters[0] == [f"A{k}" for k in range(5)]
    assert report.cluster_of("B3") == 1


def test_cluster_importance_sums_to_total(fitted):
    model, X, _ = fitted
    report = ClusteredImportance().compute(model, X)
    total = model.booster_.feature_importance("gain").sum()
    assert report.total_importance == pytest.approx(total)
    assert sum(report.cluster_share.values()) == pytest.approx(1.0)
    for cid, members in report.clusters.items():
        shares = [report.feature_importance[f][1] for f in members]
        assert sum(shares) == pytest.approx(1.0)
        assert all(report.feature_importance[f][0] == cid for f in members)
    # Informative groups dominate; the target weights are A=3 > B=2 > C=1.
    share = report.cluster_share
    a, b, c = (share[report.cluster_of(f"{g}0")] for g in GROUPS)
    assert a > b > c
    assert sum(share[report.cluster_of(f"noise{i}")] for i in range(3)) < 0.02


def test_swapping_most_important_feature_leaves_cluster_importance_unchanged():
    # Same latent signals and target; only which copy of signal A is the clean one differs.
    X1, y = _grouped_data(noise_levels=(0.1, 0.6, 0.6, 0.6, 0.6))
    X2, y2 = _grouped_data(noise_levels=(0.6, 0.1, 0.6, 0.6, 0.6))
    np.testing.assert_allclose(y, y2)
    r1 = ClusteredImportance().compute(lgb.LGBMRegressor(**LGBM_KW).fit(X1, y), X1)
    r2 = ClusteredImportance().compute(lgb.LGBMRegressor(**LGBM_KW).fit(X2, y), X2)

    top = lambda r: max(r.clusters[r.cluster_of("A0")], key=lambda f: r.feature_importance[f][1])  # noqa: E731
    assert top(r1) == "A0" and top(r2) == "A1"
    # Individual attribution swings wildly...
    assert r1.feature_importance["A0"][1] - r2.feature_importance["A0"][1] > 0.3
    # ...while the cluster's share of the model barely moves.
    assert r1.cluster_share[r1.cluster_of("A0")] == pytest.approx(r2.cluster_share[r2.cluster_of("A0")], abs=0.05)


def test_dropping_a_correlated_feature_keeps_cluster_importance():
    X, y = _grouped_data(noise_levels=(0.1, 0.3, 0.6, 0.6, 0.6))
    full = ClusteredImportance().compute(lgb.LGBMRegressor(**LGBM_KW).fit(X, y), X)
    X_drop = X.drop(columns=["A0"])
    dropped = ClusteredImportance().compute(lgb.LGBMRegressor(**LGBM_KW).fit(X_drop, y), X_drop)

    raw_share = lambda r, f: r.raw_feature_importance[f] / r.total_importance  # noqa: E731
    # The substitute inherits the dropped feature's credit...
    assert raw_share(dropped, "A1") > 2 * raw_share(full, "A1")
    # ...but the cluster-level view is unchanged.
    assert full.cluster_share[full.cluster_of("A1")] == pytest.approx(
        dropped.cluster_share[dropped.cluster_of("A1")], abs=0.05
    )


def test_stability_static_scores():
    # 3 stable clusters on top, 3 clusters shuffled among themselves below them.
    snapshots = np.array(
        [
            [100, 80, 60, 30, 20, 10],
            [100, 80, 60, 10, 30, 20],
            [100, 80, 60, 20, 10, 30],
        ],
        dtype=float,
    )
    scores, overall = ClusteredImportance.stability(snapshots)
    assert np.allclose(scores[:3], 1.0)
    assert scores[3:].max() < scores[:3].min()
    assert scores[3:].max() <= 0.6
    # The per-cluster scores are Kendall's tau decomposed per item (no ties -> tau-a == tau-b).
    assert scores.mean() == pytest.approx(overall)

    # Identical snapshots: perfectly stable everywhere.
    same, overall_same = ClusteredImportance.stability(np.tile([5.0, 3.0, 1.0], (3, 1)))
    assert np.allclose(same, 1.0) and overall_same == pytest.approx(1.0)
    # Complete reversal: -1 everywhere.
    rev, overall_rev = ClusteredImportance.stability(np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]))
    assert np.allclose(rev, -1.0) and overall_rev == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        ClusteredImportance.stability(np.ones((1, 3)))


def test_stability_from_feature_snapshots(fitted):
    model, X, _ = fitted
    ci = ClusteredImportance()
    base = ci.compute(model, X)
    # Stable: A, B, C keep 300/200/100 split across their members. Shuffled: the
    # three noise singletons rotate 30/20/10.
    rotation = [[30, 20, 10], [10, 30, 20], [20, 10, 30]]
    snaps = []
    for t in range(3):
        snap = {}
        for g, total in zip(GROUPS, (300, 200, 100)):
            weights = np.random.default_rng(t).dirichlet(np.ones(5))  # intra-cluster split shuffles too
            snap.update({f"{g}{k}": total * w for k, w in enumerate(weights)})
        snap.update({f"noise{i}": rotation[t][i] for i in range(3)})
        snaps.append(snap)
    report = ci.compute(model, X, snapshots=snaps)
    assert report.n_snapshots == 3
    stable = [report.stability_scores[base.cluster_of(f"{g}0")] for g in GROUPS]
    shuffled = [report.stability_scores[base.cluster_of(f"noise{i}")] for i in range(3)]
    assert np.allclose(stable, 1.0)  # intra-cluster reshuffling does not affect cluster stability
    assert max(shuffled) < min(stable)
    # Array snapshots (column order) work too.
    arrays = [np.array([s[c] for c in X.columns]) for s in snaps]
    assert ci.compute(model, X, snapshots=arrays).stability_scores == report.stability_scores
    # One snapshot is not enough.
    assert ci.compute(model, X, snapshots=snaps[:1]).stability_scores is None


def test_mutual_info_catches_nonmonotonic_dependence():
    rng = np.random.default_rng(1)
    n = 1000
    x = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "x": x,
            "x_copy": x + 0.3 * rng.normal(size=n),
            "x_squared": x**2 + 0.05 * rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    y = x + x**2 + 0.1 * rng.normal(size=n)
    model = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)

    spearman = ClusteredImportance(method="spearman").compute(model, X)
    mi = ClusteredImportance(method="mutual_info").compute(model, X)
    assert _as_sets(spearman.clusters) == {frozenset({"x", "x_copy"}), frozenset({"x_squared"}), frozenset({"z"})}
    assert _as_sets(mi.clusters) == {frozenset({"x", "x_copy", "x_squared"}), frozenset({"z"})}

    dep = mi.correlation_matrix.to_numpy()
    assert np.allclose(dep, dep.T)
    assert np.all((dep >= 0) & (dep <= 1))
    assert np.allclose(np.diag(dep), 1.0)
    assert mi.correlation_matrix.loc["x", "z"] < 0.2
    assert mi.method == "mutual_info"


def test_mutual_info_falls_back_for_wide_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 501))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    model = lgb.LGBMRegressor(n_estimators=3, verbose=-1).fit(X, y)
    with pytest.warns(UserWarning, match="falling back to Spearman"):
        report = ClusteredImportance(method="mutual_info").compute(model, X)
    assert report.method == "spearman"
    assert len(report.feature_importance) == 501


def test_pearson_method(fitted):
    model, X, _ = fitted
    report = ClusteredImportance(method="pearson").compute(model, X)
    assert report.n_clusters == 6
    assert report.method == "pearson"


def test_correlation_matrix_is_cached(fitted):
    model, X, _ = fitted
    ci = ClusteredImportance()
    ci.compute(model, X)
    assert (ci.cache_hits_, ci.cache_misses_) == (0, 1)
    ci.compute(model, X)
    ci.compute(model, X.copy())  # same content, different object -> still a hit
    assert (ci.cache_hits_, ci.cache_misses_) == (2, 1)
    ci.compute(model, X.iloc[:2000])
    assert ci.cache_misses_ == 2
    ci.clear_cache()
    ci.compute(model, X)
    assert ci.cache_misses_ == 3


def test_accepts_booster_and_sklearn_model(fitted):
    model, X, _ = fitted
    ci = ClusteredImportance()
    from_sklearn = ci.compute(model, X)
    from_booster = ci.compute(model.booster_, X)
    assert from_sklearn.clusters == from_booster.clusters
    assert from_sklearn.cluster_importance == from_booster.cluster_importance
    # Plain arrays: booster feature names are used.
    from_array = ci.compute(model.booster_, X.to_numpy())
    assert set(from_array.feature_importance) == set(X.columns)
    # Reordered DataFrame columns are aligned by name.
    shuffled = ci.compute(model, X[X.columns[::-1]])
    assert shuffled.cluster_importance == pytest.approx(from_sklearn.cluster_importance)
    with pytest.raises(TypeError):
        ci.compute(object(), X)


def test_threshold_extremes(fitted):
    model, X, _ = fitted
    everything_separate = ClusteredImportance(distance_threshold=0.0).compute(model, X)
    assert everything_separate.n_clusters == X.shape[1]
    one_blob = ClusteredImportance(distance_threshold=1.0).compute(model, X)
    assert one_blob.n_clusters == 1


def test_permutation_importance_separates_signal_from_noise(fitted):
    model, X, y = fitted
    report = ClusteredImportance(n_repeats=2).compute(model, X, y)
    perm = report.cluster_permutation_importance
    assert report.permutation_metric == "rmse"
    signal = [perm[report.cluster_of(f"{g}0")] for g in GROUPS]
    noise = [perm[report.cluster_of(f"noise{i}")] for i in range(3)]
    assert min(signal) > 0.3
    assert max(abs(v) for v in noise) < 0.05
    assert signal[0] > signal[1] > signal[2]


def test_constant_and_missing_features():
    rng = np.random.default_rng(3)
    n = 1000
    s = rng.normal(size=n)
    X = pd.DataFrame({"a": s, "b": s + 0.2 * rng.normal(size=n), "const": 1.0, "c": rng.normal(size=n)})
    X.loc[rng.choice(n, 100, replace=False), "b"] = np.nan
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(X, s + X["c"])
    report = ClusteredImportance().compute(model, X)
    assert _as_sets(report.clusters) == {frozenset({"a", "b"}), frozenset({"const"}), frozenset({"c"})}
    assert report.cluster_importance[report.cluster_of("const")] == 0.0
    assert report.feature_importance["const"][1] == 1.0


def test_report_rendering(fitted, capsys):
    model, X, y = fitted
    snaps = [dict(zip(X.columns, rng_row)) for rng_row in np.random.default_rng(0).random((3, X.shape[1]))]
    report = ClusteredImportance().compute(model, X, y, snapshots=snaps)
    assert isinstance(report, ClusteredImportanceReport)
    report.summary()
    out = capsys.readouterr().out
    assert "Clustered feature importance" in out and "stability" in out and "A0" in out
    frame = report.to_frame()
    assert list(frame.columns) == ["importance", "share", "n_features", "stability", "permutation_importance", "features"]
    assert frame["share"].is_monotonic_decreasing
    assert report.feature_frame().shape == (X.shape[1], 3)
    assert report.correlation_matrix.shape == (X.shape[1], X.shape[1])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ClusteredImportance(method="kendall")
    with pytest.raises(ValueError):
        ClusteredImportance(linkage="ward")
    with pytest.raises(ValueError):
        ClusteredImportance(distance_threshold=1.5)
