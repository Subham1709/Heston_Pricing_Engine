import numpy as np
import pandas as pd
import pytest

from heston.calibration import calibrate, prepare_quotes
from heston.fourier import cos_price
from heston.model import HestonParams


def synthetic_chain(true, S=100.0, r=0.03, q=0.01):
    rows = []
    for T in [0.1, 0.25, 0.5, 1.0, 2.0]:
        for K in np.arange(70, 131, 5.0):
            for call in (True, False):
                m = cos_price(S, K, T, r, true, q, call=call)
                rows.append(dict(expiry=f"{T}", T=T, K=K, call=call, bid=m * 0.999, ask=m * 1.001))
    return pd.DataFrame(rows)


def test_forward_and_rates_recovered():
    true = HestonParams(0.03, 2.5, 0.05, 0.6, -0.7)
    q = prepare_quotes(synthetic_chain(true), 100.0)
    assert np.allclose(q.r, 0.03, atol=1e-6) and np.allclose(q.q, 0.01, atol=1e-6)
    assert (q.call == (q.K >= q.F)).all()  # OTM only


def test_calibration_recovers_parameters():
    true = HestonParams(0.03, 2.5, 0.05, 0.6, -0.7)
    q = prepare_quotes(synthetic_chain(true), 100.0)
    p, info = calibrate(q, 100.0)
    assert np.allclose(p.as_array(), true.as_array(), rtol=1e-3)
    assert info["rmse_iv"] < 1e-6
