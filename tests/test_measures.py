
import numpy as np
import pandas as pd
import pytest
import torch

from latent_switch.measures import entropy_bits, make_controls, measure
from latent_switch.stats import mean_ci


@pytest.mark.parametrize("n", [2, 5, 98])
def test_make_controls_is_derangement(n):
    ctrl = make_controls(n, seed=1)
    assert (ctrl != np.arange(n)).all()                 # nobody is their own control
    assert sorted(ctrl) == list(range(n))               # each row used exactly once


def test_entropy_uniform_and_with_zeros():
    assert torch.allclose(entropy_bits(torch.full((8,), 1 / 8)), torch.tensor(3.0))
    assert entropy_bits(torch.tensor([1.0, 0.0])).item() == 0.0     # no nan from 0·log 0


def _row(tgt_id, tgt_prefix, lat_id, lat_prefix):
    return pd.Series({"zh_id": tgt_id, "zh_prefix": tgt_prefix, "en_id": lat_id, "en_prefix": lat_prefix})


def test_measure_symmetric_overlap_and_ranks():
    probs = torch.tensor([[0.05, 0.10, 0.20, 0.15, 0.30, 0.20]])     # 1 layer, 6 tokens
    row = _row(0, [0], 1, [1, 2])                                     # matched en prefix {1, 2}
    ctrl = _row(5, [5], 3, [2, 3])                                    # control en prefix {2, 3}; 2 is shared
    m = measure(probs, row, ctrl, {"tgt": "zh", "lat": "en"})
    assert m["lat_prefix"].item() == pytest.approx(0.10)              # shared id 2 removed from matched...
    assert m["lat_ctrl_prefix"].item() == pytest.approx(0.15)         # ...and from control
    assert m["rank_tgt"].item() == 5 and m["rank_lat"].item() == 4    # number of tokens with higher prob
    assert m["rank_lat_ctrl"].item() == 3
    assert m["entropy"].shape == (1,)


def test_measure_first_token_collision_gives_nan():
    probs = torch.full((2, 4), 0.25)
    m = measure(probs, _row(0, [0], 1, [1]), _row(2, [2], 1, [1]), {"tgt": "zh", "lat": "en"})
    assert torch.isnan(m["rank_lat_ctrl"]).all() and torch.isnan(m["lat_ctrl_first"]).all()


def test_mean_ci_ignores_nan():
    x = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 2.0]])
    mean, lo, hi = mean_ci(x)
    assert mean.tolist() == [3.0, 2.0]
    assert lo[1] == hi[1] == 2.0                                      # zero spread → zero-width CI
    assert lo[0] < 3.0 < hi[0]
