import numpy as np
import pytest

from heston.black_scholes import bs_greeks, bs_price, implied_vol


def test_textbook_value():
    # Hull, Options Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, vol=20%, T=0.5
    assert bs_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=5e-3)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, call=False) == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("K", [60, 90, 100, 110, 160])
def test_put_call_parity(K):
    S, T, r, q, s = 100, 0.75, 0.05, 0.02, 0.3
    c = bs_price(S, K, T, r, s, q, call=True)
    p = bs_price(S, K, T, r, s, q, call=False)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("call", [True, False])
def test_greeks_match_finite_differences(call):
    S, K, T, r, q, s = 100, 105, 0.5, 0.03, 0.01, 0.25
    g = bs_greeks(S, K, T, r, s, q, call)
    h = 1e-4
    f = lambda **kw: bs_price(**{**dict(S=S, K=K, T=T, r=r, sigma=s, q=q, call=call), **kw})
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=s + h) - f(sigma=s - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    # theta is dV/dt = -dV/dT
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("K", [50, 80, 100, 125, 200])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_roundtrip(K, sigma, call):
    S, T, r = 100, 1.0, 0.02
    price = bs_price(S, K, T, r, sigma, call=call)
    intrinsic = max(S - K * np.exp(-r * T), 0) if call else max(K * np.exp(-r * T) - S, 0)
    if price - intrinsic < 1e-8:  # (almost) no time value left, so no vol information
        pytest.skip("time value underflows")
    assert implied_vol(price, S, K, T, r, call=call) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage():
    assert np.isnan(implied_vol(150.0, 100, 100, 1.0, 0.0))  # call above spot
    assert np.isnan(implied_vol(-1.0, 100, 100, 1.0, 0.0))
