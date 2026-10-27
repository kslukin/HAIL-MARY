import pandas as pd
import pytest

from tsagent.analysis.selection import selection_vs_fixed


def make_mase(a, b, windows, series):
    index = pd.MultiIndex.from_product([windows, series], names=["window", "unique_id"])
    return pd.DataFrame({"a": a, "b": b}, index=index)


def test_selection_uses_only_the_previous_window():
    # Model a is better in window 0, model b in window 1: the choice for window 1 must be a.
    mase = make_mase(a=[1.0, 3.0], b=[2.0, 1.0], windows=[0, 1], series=["s1"])
    out = selection_vs_fixed(mase)
    assert len(out) == 1
    assert out.loc[0, "chosen_model"] == "a"
    assert out.loc[0, "selected"] == pytest.approx(3.0)  # not the hindsight value of 1.0
    assert out.loc[0, "hindsight_best"] == pytest.approx(1.0)


def test_selection_beats_fixed_policies_when_the_best_model_differs_by_series():
    # Series s1 is always better with a, series s2 always with b.
    mase = make_mase(a=[1.0, 2.0] * 3, b=[2.0, 1.0] * 3, windows=[0, 1, 2], series=["s1", "s2"])
    out = selection_vs_fixed(mase)
    assert set(out["window"]) == {1, 2}
    assert out["selected"].mean() == pytest.approx(1.0)
    assert out["a"].mean() == pytest.approx(1.5)
    assert out["b"].mean() == pytest.approx(1.5)


def test_longer_lookback_changes_the_choice():
    # Series s1: a wins window 0 clearly, b wins window 1 slightly.
    # lookback=1 looks only at window 1 -> b; lookback=2 averages both windows -> a.
    mase = make_mase(
        a=[1.0, 4.0, 7.0], b=[3.0, 3.0, 8.0], windows=[0, 1, 2], series=["s1"]
    )
    short = selection_vs_fixed(mase, lookback=1)
    long = selection_vs_fixed(mase, lookback=2)
    assert short.loc[short["window"] == 2, "chosen_model"].item() == "b"
    assert long.loc[long["window"] == 2, "chosen_model"].item() == "a"
