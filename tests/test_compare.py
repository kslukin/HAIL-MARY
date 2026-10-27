import numpy as np
import pandas as pd
import pytest

from tsagent.analysis.compare import bootstrap_mean_ci, paired_comparison, segment_features


def test_bootstrap_interval_of_constant_values_is_degenerate():
    mean, low, high = bootstrap_mean_ci(np.full(50, 0.3), n_boot=100)
    assert mean == pytest.approx(0.3)
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.3)


def test_paired_comparison_detects_a_consistent_difference():
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product(
        [[0, 1], [f"s{i}" for i in range(60)]], names=["window", "unique_id"]
    )
    base = rng.uniform(0.5, 1.5, size=len(index))
    mase = pd.DataFrame({"a": base - 0.1, "b": base, "c": base}, index=index)

    out = paired_comparison(mase, n_boot=300).set_index(["A", "B"])
    assert out.loc[("a", "b"), "mean_A_minus_B"] == pytest.approx(-0.1)
    assert out.loc[("a", "b"), "ci_excludes_zero"]  # a is better in every single series
    assert not out.loc[("b", "c"), "ci_excludes_zero"]  # b and c are identical


def make_long():
    ds = pd.date_range("2020-01-01", periods=450)
    full = pd.DataFrame({"unique_id": "full", "ds": ds, "y": 1.0})
    late = pd.DataFrame({"unique_id": "late", "ds": ds[100:], "y": 0.0})  # starts late, all zeros
    return pd.concat([full, late], ignore_index=True)


def test_segment_features_describe_series_at_the_cutoff():
    feats = segment_features(make_long(), horizon=7, n_windows=1, min_history=14)
    assert feats.loc[(0, "full"), "length"] == "long (>=365 days)"
    assert feats.loc[(0, "late"), "length"] == "short (<365 days)"
    assert feats.loc[(0, "full"), "intermittency"] == "regular"
    assert feats.loc[(0, "late"), "intermittency"] == "intermittent (>=50% zero days)"


def test_segment_features_ignore_data_after_the_cutoff():
    base = make_long()
    changed = base.copy()
    test_period = changed["ds"] > changed["ds"].max() - pd.Timedelta(days=7)
    changed.loc[test_period, "y"] = 999.0  # rewrite the part that is the test period
    pd.testing.assert_frame_equal(
        segment_features(base, horizon=7, n_windows=1, min_history=14),
        segment_features(changed, horizon=7, n_windows=1, min_history=14),
    )
