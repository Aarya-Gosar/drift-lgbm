import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from drift_lgbm.distribution_tracker import DistributionTracker, DriftReport


def _normal_frame(rng, n, n_features, shift=None):
    shift = shift or {}
    data = {f"f{j}": rng.normal(shift.get(j, 0.0), 1.0, n) for j in range(n_features)}
    return pd.DataFrame(data)


def test_first_update_establishes_reference():
    rng = np.random.default_rng(0)
    tracker = DistributionTracker()
    report = tracker.update(_normal_frame(rng, 1000, 3))
    assert isinstance(report, DriftReport)
    assert report.is_reference
    assert report.drifted_features == []
    assert not tracker.should_rebin()
    assert tracker.feature_names_ == ["f0", "f1", "f2"]


def test_detects_mean_shift_only_on_shifted_features():
    rng = np.random.default_rng(1)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 3000, 6))
    report = tracker.update(_normal_frame(rng, 3000, 6, shift={1: 0.5, 4: 1.0}))

    assert set(report.drifted_features) == {"f1", "f4"}
    # Ordered by severity: the 1.0-sigma shift is more drifted than the 0.5-sigma one.
    assert report.drifted_features == ["f4", "f1"]
    assert report.mean_shift["f4"] == pytest.approx(1.0, abs=0.1)
    assert report.mean_shift["f1"] == pytest.approx(0.5, abs=0.1)
    assert report.ks_statistic["f4"] > report.ks_statistic["f1"] > report.ks_statistic["f0"]
    for stable in ["f0", "f2", "f3", "f5"]:
        assert report.p_value[stable] > 0.01


def test_stable_features_rarely_flagged():
    # Under the null the grid KS test is conservative: false positives stay at or below nominal.
    rng = np.random.default_rng(2)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 2000, 50))
    flagged = sum(tracker.update(_normal_frame(rng, 2000, 50)).n_drifted for _ in range(10))
    assert flagged / 500 <= 0.02


def test_grid_ks_statistic_matches_scipy():
    rng = np.random.default_rng(3)
    for shift, scale in [(0.3, 1.2), (0.05, 1.0), (0.0, 1.0)]:
        ref = rng.normal(0, 1, 5000)
        new = rng.normal(shift, scale, 4000)
        tracker = DistributionTracker(n_bins=256)
        tracker.update(ref)
        report = tracker.update(new)
        exact = stats.ks_2samp(ref, new, method="asymp")
        # The grid statistic never exceeds the exact one and is within the grid
        # resolution of it, so its p-value is conservative (up to the small
        # difference between the Stephens approximation and scipy's kstwo).
        assert report.ks_statistic[0] <= exact.statistic + 1e-12
        assert exact.statistic - report.ks_statistic[0] < 2.0 / 256
        assert report.p_value[0] >= 0.95 * exact.pvalue


def test_out_of_range_shift_detected_in_both_directions():
    rng = np.random.default_rng(4)
    tracker = DistributionTracker()
    tracker.update(rng.normal(0, 1, (1000, 2)))
    below = np.column_stack([rng.normal(-10, 1, 1000), rng.normal(10, 1, 1000)])
    report = tracker.update(below)
    assert set(report.drifted_features) == {0, 1}
    assert report.ks_statistic[0] == pytest.approx(1.0)
    assert report.ks_statistic[1] == pytest.approx(1.0)
    assert report.out_of_range_fraction[0] == pytest.approx(1.0)
    assert report.out_of_range_fraction[1] == pytest.approx(1.0)


def test_should_rebin_fraction_threshold():
    rng = np.random.default_rng(5)
    tracker = DistributionTracker(drift_fraction=0.3)
    tracker.update(_normal_frame(rng, 2000, 10))

    # 3/10 drifted: exactly 30% is not *more* than 30%.
    tracker.update(_normal_frame(rng, 2000, 10, shift={0: 2, 1: 2, 2: 2}))
    assert tracker.last_report_.n_drifted == 3
    assert not tracker.should_rebin()

    # 4/10 drifted -> rebin.
    report = tracker.update(_normal_frame(rng, 2000, 10, shift={0: 2, 1: 2, 2: 2, 3: 2}))
    assert report.n_drifted == 4
    assert tracker.should_rebin()
    assert report.rebin_recommended

    # A stricter fraction threshold changes the decision for the same data.
    strict = DistributionTracker(drift_fraction=0.5)
    strict.update(_normal_frame(rng, 2000, 10))
    strict.update(_normal_frame(rng, 2000, 10, shift={0: 2, 1: 2, 2: 2, 3: 2}))
    assert not strict.should_rebin()


def test_should_rebin_uses_explicit_report():
    rng = np.random.default_rng(6)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 1000, 2))
    drifted = tracker.update(_normal_frame(rng, 1000, 2, shift={0: 3, 1: 3}))
    tracker.update(_normal_frame(rng, 1000, 2, shift={0: 3, 1: 3}))  # reference unchanged -> still drifted
    assert tracker.should_rebin(drifted)


def test_reset_moves_reference():
    rng = np.random.default_rng(7)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 2000, 3))
    shifted = _normal_frame(rng, 2000, 3, shift={0: 3, 1: 3, 2: 3})
    assert tracker.update(shifted).n_drifted == 3
    tracker.reset(shifted)
    report = tracker.update(_normal_frame(rng, 2000, 3, shift={0: 3, 1: 3, 2: 3}))
    assert report.n_drifted == 0
    assert tracker.feature_stats("f0").reference_mean == pytest.approx(3.0, abs=0.1)


def test_serialization_round_trip():
    rng = np.random.default_rng(8)
    tracker = DistributionTracker(n_bins=64)
    tracker.update(_normal_frame(rng, 1500, 4))
    tracker.update(_normal_frame(rng, 1500, 4, shift={2: 1}))

    clone = pickle.loads(pickle.dumps(tracker))
    assert clone.feature_names_ == tracker.feature_names_
    assert clone.n_batches_seen_ == tracker.n_batches_seen_
    np.testing.assert_array_equal(clone.reference_quantiles("f2"), tracker.reference_quantiles("f2"))

    nxt = _normal_frame(rng, 1500, 4, shift={0: 1, 3: 1})
    r1, r2 = tracker.update(nxt), clone.update(nxt)
    assert r1.drifted_features == r2.drifted_features
    assert r1.p_value == r2.p_value
    assert r1.ks_statistic == r2.ks_statistic


def test_single_feature_numpy_1d_input():
    rng = np.random.default_rng(9)
    tracker = DistributionTracker()
    tracker.update(rng.normal(0, 1, 1000))
    assert tracker.feature_names_ == [0]
    report = tracker.update(rng.normal(2, 1, 1000))
    assert report.drifted_features == [0]
    assert tracker.should_rebin()  # 1/1 features drifted


def test_constant_feature():
    rng = np.random.default_rng(10)
    ref = np.column_stack([np.full(1000, 5.0), rng.normal(0, 1, 1000)])
    tracker = DistributionTracker()
    tracker.update(ref)
    stats_ = tracker.feature_stats(0)
    assert stats_.reference_std == 0.0
    assert stats_.n_bins == 1

    same = np.column_stack([np.full(800, 5.0), rng.normal(0, 1, 800)])
    report = tracker.update(same)
    assert report.p_value[0] == 1.0
    assert 0 not in report.drifted_features
    assert np.isfinite(report.mean_shift[0])

    lower = np.column_stack([np.full(800, 4.0), rng.normal(0, 1, 800)])
    report = tracker.update(lower)
    assert 0 in report.drifted_features
    assert report.mean_shift[0] == -np.inf

    spread = np.column_stack([rng.normal(5.0, 1.0, 800), rng.normal(0, 1, 800)])
    assert 0 in tracker.update(spread).drifted_features


def test_nan_values_are_ignored():
    rng = np.random.default_rng(11)
    ref = _normal_frame(rng, 2000, 3)
    ref.iloc[rng.choice(2000, 600, replace=False), 0] = np.nan
    tracker = DistributionTracker()
    tracker.update(ref)
    st = tracker.feature_stats("f0")
    assert st.reference_count == 1400
    assert st.reference_missing == 600
    assert np.isfinite(st.reference_mean)
    assert np.all(np.isfinite(tracker.reference_quantiles("f0")))

    # Same distribution, different missingness pattern (plus an inf) -> no drift.
    new = _normal_frame(rng, 2000, 3)
    new.iloc[:900, 0] = np.nan
    new.iloc[0, 1] = np.inf
    report = tracker.update(new)
    assert report.drifted_features == []
    assert tracker.feature_stats("f0").running_missing == 1500
    assert tracker.feature_stats("f1").running_missing == 1


def test_all_nan_feature():
    rng = np.random.default_rng(12)
    ref = np.column_stack([np.full(500, np.nan), rng.normal(0, 1, 500)])
    tracker = DistributionTracker()
    tracker.update(ref)
    report = tracker.update(np.column_stack([rng.normal(0, 1, 500), rng.normal(0, 1, 500)]))
    assert report.p_value[0] == 1.0
    assert 0 not in report.drifted_features
    report = tracker.update(np.column_stack([np.full(500, np.nan), rng.normal(0, 1, 500)]))
    assert report.p_value[0] == 1.0


def test_small_batch_loosens_threshold():
    tracker = DistributionTracker(drift_threshold=0.01, small_batch_size=500, max_small_batch_threshold=0.1)
    assert tracker.effective_threshold(5000) == 0.01
    assert tracker.effective_threshold(500) == 0.01
    assert tracker.effective_threshold(250) == pytest.approx(0.02)
    assert tracker.effective_threshold(10) == pytest.approx(0.1)
    assert DistributionTracker(small_batch_size=0).effective_threshold(10) == 0.01

    rng = np.random.default_rng(13)
    tracker.update(rng.normal(0, 1, (5000, 2)))
    report = tracker.update(rng.normal(0, 1, (100, 2)))
    assert report.p_value_threshold == pytest.approx(0.05)


def test_min_ks_statistic_filters_tiny_shifts():
    rng = np.random.default_rng(14)
    ref, new = rng.normal(0, 1, 50_000), rng.normal(0.05, 1, 50_000)
    loose = DistributionTracker()
    loose.update(ref)
    assert loose.update(new).drifted_features == [0]  # statistically significant...
    strict = DistributionTracker(min_ks_statistic=0.05)
    strict.update(ref)
    assert strict.update(new).drifted_features == []  # ...but practically tiny


def test_running_histogram_accumulates():
    rng = np.random.default_rng(15)
    tracker = DistributionTracker(n_bins=16)
    tracker.update(rng.uniform(0, 1, 1600))
    edges, ref_counts = tracker.reference_histogram(0)
    assert len(edges) == 17
    assert ref_counts.sum() == 1600
    assert np.all(np.abs(ref_counts - 100) <= 1)  # quantile bins are equal-mass

    tracker.update(rng.uniform(0.5, 1.5, 1000))
    _, run_counts, under, over = tracker.running_histogram(0)
    assert run_counts.sum() + under + over == 2600
    assert under == 0
    assert over == pytest.approx(500, abs=60)
    assert tracker.feature_stats(0).running_max > 1.4


def test_dataframe_column_alignment():
    rng = np.random.default_rng(16)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 1000, 3))
    reordered = _normal_frame(rng, 1000, 3, shift={2: 3})[["f2", "f0", "f1"]]
    assert tracker.update(reordered).drifted_features == ["f2"]
    # Plain arrays are matched by position.
    assert tracker.update(_normal_frame(rng, 1000, 3).to_numpy()).n_features == 3
    with pytest.raises(ValueError):
        tracker.update(_normal_frame(rng, 1000, 3).rename(columns={"f0": "zzz"}))
    with pytest.raises(ValueError):
        tracker.update(rng.normal(size=(100, 4)))


def test_report_rendering():
    rng = np.random.default_rng(17)
    tracker = DistributionTracker()
    tracker.update(_normal_frame(rng, 1000, 4))
    report = tracker.update(_normal_frame(rng, 1000, 4, shift={0: 2, 1: 2}))
    frame = report.to_frame()
    assert list(frame.index[:2]) == ["f0", "f1"] or list(frame.index[:2]) == ["f1", "f0"]
    assert frame["drifted"].sum() == 2
    text = report.to_string()
    assert "2/4 features drifted" in text and "RECOMMENDED" in text


def test_invalid_arguments():
    with pytest.raises(ValueError):
        DistributionTracker(n_bins=1)
    with pytest.raises(ValueError):
        DistributionTracker(drift_threshold=0)
    with pytest.raises(ValueError):
        DistributionTracker(drift_fraction=1.0)
    with pytest.raises(RuntimeError):
        DistributionTracker().feature_stats(0)
    with pytest.raises(TypeError):
        DistributionTracker().update(pd.DataFrame({"a": ["x", "y"]}))
