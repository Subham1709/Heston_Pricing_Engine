import numpy as np
import pytest

from heston.black_scholes import bs_price
from heston.fourier import carr_madan_fft, cos_price, heston_price_integral
from heston.model import HestonParams, heston_cf

# Fang & Oosterlee (2008), section 5.2: reference value 5.785155450
FO = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
FO_REF = 5.785155450
# In 't Hout & Foulon (2010), case 1 (strong negative correlation, long maturity tested)
IHF = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.9)


def test_cf_basic_properties():
    T, r, q = 1.3, 0.03, 0.01
    assert heston_cf(0.0, T, IHF, r, q) == pytest.approx(1.0)
    # martingale condition: E[S_T / S_0] = e^{(r-q)T}
    assert heston_cf(-1j, T, IHF, r, q).real == pytest.approx(np.exp((r - q) * T), rel=1e-12)


def test_cf_continuous_long_maturity():
    """The little-trap form must have no jumps in u (the original Heston form does)."""
    u = np.linspace(0, 50, 20001)
    phi = heston_cf(u, 30.0, FO)
    assert np.max(np.abs(np.diff(phi))) < 1e-2


@pytest.mark.parametrize("pricer,tol", [(heston_price_integral, 1e-7), (carr_madan_fft, 1e-6),
                                        (cos_price, 1e-6)])
def test_fang_oosterlee_reference(pricer, tol):
    assert pricer(100, 100, 1.0, 0.0, FO) == pytest.approx(FO_REF, abs=tol)


@pytest.mark.parametrize("T", [0.01, 0.25, 1.0, 10.0])
def test_cos_matches_integral_across_strikes(T):
    K = np.array([50.0, 75, 100, 130, 200])
    for call in (True, False):
        ref = np.array([heston_price_integral(100, k, T, 0.025, IHF, call=call) for k in K])
        assert np.allclose(cos_price(100, K, T, 0.025, IHF, call=call), ref, atol=1e-6)


def test_black_scholes_limit():
    """sigma -> 0 and v0 = theta: variance stays at theta, so Heston = BS with vol sqrt(theta)."""
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0)
    K = np.array([70.0, 100, 140])
    assert np.allclose(cos_price(100, K, 0.7, 0.03, p, q=0.01),
                       bs_price(100, K, 0.7, 0.03, 0.2, q=0.01), atol=1e-6)


def test_no_arbitrage_shape():
    """Call prices must be decreasing and convex in strike."""
    K = np.linspace(60, 160, 101)
    c = cos_price(100, K, 0.5, 0.02, FO)
    assert np.all(np.diff(c) < 0)
    assert np.all(np.diff(c, 2) > -1e-10)
