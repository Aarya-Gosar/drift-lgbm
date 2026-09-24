import numpy as np
import pandas as pd
import pytest

from drift_lgbm.distribution_tracker import DistributionTracker
from drift_lgbm.testing import generate_financial_data


def test_shapes_names_and_metadata():
    X, y, meta = generate_financial_data(n_samples=2000)
    assert X.shape == (2000, 3 * 5 + 10 + 10)
    assert y.shape == (2000,) and y.name == "target"
    assert X.columns.is_unique
    assert list(meta["groups"]) == ["momentum", "volatility", "value"]
    assert meta["groups"]["momentum"] == ["momentum_5d", "momentum_10d", "momentum_21d", "momentum_63d", "momentum_126d"]
    assert "vol_realized_21d" in X.columns and "earnings_surprise" in X.columns
    assert len(meta["informative_features"]) == 10 and len(meta["noise_features"]) == 10
    all_listed = sum(meta["groups"].values(), []) + meta["informative_features"] + meta["noise_features"]
    assert all_listed == list(X.columns)
    assert meta["regime_change_at"] is None and meta["regime_shift"] is None and meta["shifted_features"] == []
    assert set(meta["feature_to_group"]) == set(X.columns)
    assert X.notna().all().all()


def test_correlation_structure():
    X, _, meta = generate_financial_data(n_samples=20_000, random_state=1)
    corr = X.corr(method="spearman").abs()
    groups = meta["groups"]
    for names in groups.values():
        within = corr.loc[names, names].to_numpy()[np.triu_indices(len(names), 1)]
        assert within.min() > 0.5  # clusters at the default 0.5 distance threshold
    group_names = list(groups)
    for i, a in enumerate(group_names):
        for b in group_names[i + 1 :]:
            assert corr.loc[groups[a], groups[b]].to_numpy().max() < 0.05
    for f in meta["noise_features"]:
        others = corr.loc[f].drop(f)
        assert others.max() < 0.05
    # Some copies are cleaner than others: noise levels vary within each group.
    for names in groups.values():
        sigmas = [meta["feature_noise_levels"][f] for f in names]
        assert min(sigmas) == pytest.approx(0.2) and max(sigmas) == pytest.approx(0.8)
        assert len(set(sigmas)) == len(names)


def test_target_structure():
    X, y, meta = generate_financial_data(n_samples=20_000, target_noise=0.1, random_state=2)
    corr_with_y = X.corrwith(y).abs()
    assert corr_with_y[meta["noise_features"]].max() < 0.05
    assert all(abs(meta["coefficients"][g]) >= 0.5 for g in meta["groups"])
    # Cleaner copies track the target better than noisier ones within a family.
    names = meta["groups"]["momentum"]
    cleanest = min(names, key=lambda f: meta["feature_noise_levels"][f])
    noisiest = max(names, key=lambda f: meta["feature_noise_levels"][f])
    assert corr_with_y[cleanest] > corr_with_y[noisiest]
    # Noise-to-signal ratio: y minus the true signal has std = target_noise * signal std.
    signal_std = np.sqrt(sum(c**2 for c in meta["coefficients"].values()))
    assert meta["target_noise_std"] == pytest.approx(0.1 * signal_std)
    # Stand-alone features are the signals themselves; what remains is the latent
    # group signals plus target noise, with the variance the coefficients imply.
    stand_alone = X[meta["informative_features"]].to_numpy() @ np.array(
        [meta["coefficients"][f] for f in meta["informative_features"]]
    )
    expected_var = sum(meta["coefficients"][g] ** 2 for g in meta["groups"]) + meta["target_noise_std"] ** 2
    assert np.var(y - stand_alone) == pytest.approx(expected_var, rel=0.05)


def test_uninformative_groups():
    X, y, meta = generate_financial_data(n_samples=5000, n_correlated_groups=4, n_informative_groups=2, random_state=3)
    assert meta["informative_groups"] == ["momentum", "volatility"]
    assert meta["uninformative_groups"] == ["value", "liquidity"]
    assert meta["coefficients"]["value"] == 0.0 and meta["coefficients"]["liquidity"] == 0.0
    corr_with_y = X.corrwith(y).abs()
    assert corr_with_y[meta["groups"]["value"]].max() < 0.05


def test_regime_change_shifts_signals_not_noise():
    X, y, meta = generate_financial_data(
        n_samples=20_000, regime_change_at=10_000, regime_shift_magnitude=2.0, random_state=4
    )
    pre, post = X.iloc[:10_000], X.iloc[10_000:]
    shift = meta["regime_shift"]
    for family, names in meta["groups"].items():
        expected = shift[family]["mean_shift"]
        assert abs(expected) == pytest.approx(2.0)
        for f in names:
            assert post[f].mean() - pre[f].mean() == pytest.approx(expected, abs=0.1)
    for f in meta["informative_features"]:
        assert post[f].mean() - pre[f].mean() == pytest.approx(shift[f]["mean_shift"], abs=0.1)
        assert post[f].std() / pre[f].std() == pytest.approx(1.5, abs=0.05)  # 1 + magnitude / 4
    for f in meta["noise_features"]:
        assert post[f].mean() - pre[f].mean() == pytest.approx(0.0, abs=0.06)
    assert set(meta["shifted_features"]) == set(sum(meta["groups"].values(), [])) | set(meta["informative_features"])

    # The drift tracker sees exactly the shifted features.
    tracker = DistributionTracker()
    tracker.update(pre)
    report = tracker.update(post)
    assert set(report.drifted_features) == set(meta["shifted_features"])
    assert tracker.should_rebin()

    # Without a regime change the two halves look alike.
    X_flat, _, _ = generate_financial_data(n_samples=20_000, random_state=4)
    tracker = DistributionTracker()
    tracker.update(X_flat.iloc[:10_000])
    assert not tracker.update(X_flat.iloc[10_000:]).rebin_recommended


def test_reproducibility():
    a = generate_financial_data(n_samples=500, random_state=7)
    b = generate_financial_data(n_samples=500, random_state=7)
    c = generate_financial_data(n_samples=500, random_state=8)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_series_equal(a[1], b[1])
    assert a[2] == b[2]
    assert not a[0].equals(c[0])


def test_many_groups_and_wide_groups_get_generic_names():
    X, _, meta = generate_financial_data(
        n_samples=100, n_correlated_groups=12, features_per_group=8, n_informative=20, n_noise=3
    )
    assert X.shape[1] == 12 * 8 + 20 + 3
    assert X.columns.is_unique
    assert meta["groups"]["momentum"][-2:] == ["momentum_6", "momentum_7"]
    assert "factor11" in meta["groups"] and meta["groups"]["factor11"][0] == "factor11_0"
    assert meta["informative_features"][-1] == "alpha_signal_19"


def test_degenerate_configurations():
    X, y, meta = generate_financial_data(n_samples=100, n_informative=0, n_correlated_groups=0, n_noise=4)
    assert X.shape == (100, 4)
    assert np.all(y == 0)
    X, _, _ = generate_financial_data(n_samples=100, features_per_group=1, n_noise=0, n_informative=0)
    assert X.shape == (100, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(regime_change_at=0),
        dict(regime_change_at=10_000),
        dict(n_noise=-1),
        dict(n_informative_groups=5),
        dict(target_noise=-0.1),
        dict(group_noise_range=(0.5, 0.2)),
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_financial_data(n_samples=10_000, **kwargs)


def test_default_output_unchanged_by_concept_option():
    a = generate_financial_data(n_samples=3000, regime_change_at=1500, random_state=9)
    b = generate_financial_data(n_samples=3000, regime_change_at=1500, random_state=9, regime_concept_shift=0.0)
    pd.testing.assert_series_equal(a[1], b[1])
    assert a[2]["flipped_signals"] == [] and a[2]["coefficients"] == a[2]["coefficients_after_regime"]


def test_concept_shift_flips_factor_premia():
    X, y, meta = generate_financial_data(
        n_samples=20_000, regime_change_at=10_000, regime_concept_shift=0.5, random_state=10
    )
    active = [s for s, c in meta["coefficients"].items() if c != 0]
    assert len(meta["flipped_signals"]) == round(0.5 * len(active))
    for signal in meta["flipped_signals"]:
        assert meta["coefficients_after_regime"][signal] == -meta["coefficients"][signal]
    # A flipped stand-alone feature's relation with the target reverses after the change.
    flipped_features = [s for s in meta["flipped_signals"] if s in meta["informative_features"]]
    kept_features = [s for s in meta["informative_features"] if s not in meta["flipped_signals"]]
    f = flipped_features[0]
    pre_corr = np.corrcoef(X[f].iloc[:10_000], y.iloc[:10_000])[0, 1]
    post_corr = np.corrcoef(X[f].iloc[10_000:], y.iloc[10_000:])[0, 1]
    assert np.sign(pre_corr) == -np.sign(post_corr)
    g = kept_features[0]
    assert np.sign(np.corrcoef(X[g].iloc[:10_000], y.iloc[:10_000])[0, 1]) == np.sign(
        np.corrcoef(X[g].iloc[10_000:], y.iloc[10_000:])[0, 1]
    )
    with pytest.raises(ValueError):
        generate_financial_data(n_samples=100, regime_concept_shift=0.5)
