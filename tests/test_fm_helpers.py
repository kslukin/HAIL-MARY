import pandas as pd
import pytest

from tsagent.models.base import finalize_quantiles
from tsagent.models.timesfm_model import TimesFM3Forecaster


def test_decile_index_maps_levels_to_columns():
    assert [TimesFM3Forecaster.decile_index(q) for q in (0.1, 0.5, 0.9)] == [0, 4, 8]


@pytest.mark.parametrize("q", [0.05, 0.25, 0.95])
def test_decile_index_rejects_levels_that_are_not_deciles(q):
    with pytest.raises(ValueError):
        TimesFM3Forecaster.decile_index(q)


def test_finalize_quantiles_sorts_rows_and_clips_at_zero():
    frame = pd.DataFrame({"q0.1": [3.0, -1.0], "q0.5": [2.0, 1.0], "q0.9": [4.0, 5.0]})
    out = finalize_quantiles(frame, [0.1, 0.5, 0.9])
    assert out.loc[0].tolist() == [2.0, 3.0, 4.0]  # crossing quantiles are put in order
    assert out.loc[1].tolist() == [0.0, 1.0, 5.0]  # negative value clipped to zero
