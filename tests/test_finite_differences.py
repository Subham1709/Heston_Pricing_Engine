import numpy as np
import pytest

from heston.black_scholes import bs_price
from heston.fd1d import bs_fd
from heston.fd_adi import heston_adi
from heston.fourier import cos_price
from heston.model import HestonParams

IHF = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.9)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_crank_nicolson_matches_bs(call, K):
    ref = bs_price(100, K, 1.0, 0.05, 0.2, 0.01, call)
    assert bs_fd(100, K, 1.0, 0.05, 0.2, 0.01, call, n_space=400, n_time=200) == pytest.approx(ref, abs=2e-3)


def test_crank_nicolson_second_order_in_space():
    ref = bs_price(100, 100, 1.0, 0.05, 0.2)
    e1 = abs(bs_fd(100, 100, 1.0, 0.05, 0.2, n_space=100, n_time=100) - ref)
    e2 = abs(bs_fd(100, 100, 1.0, 0.05, 0.2, n_space=200, n_time=200) - ref)
    assert 3.0 < e1 / e2 < 5.0  # halving h should divide the error by ~4


def test_american_put_value():
    """S = K = 100, r = 5%, vol = 20%, T = 1: American put = 6.0904 (binomial, 10k steps)."""
    am = bs_fd(100, 100, 1.0, 0.05, 0.2, call=False, american=True, n_space=400, n_time=200)
    eu = bs_price(100, 100, 1.0, 0.05, 0.2, call=False)
    assert am == pytest.approx(6.0904, abs=3e-3)
    assert am > eu  # early exercise premium


@pytest.mark.parametrize("scheme", ["cs", "hv"])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_adi_matches_cos(scheme, K):
    ref = cos_price(100, K, 1.0, 0.025, IHF)
    assert heston_adi(100, K, 1.0, 0.025, IHF, scheme=scheme, n_S=200, n_v=100,
                      n_time=50) == pytest.approx(ref, abs=5e-3)


def test_douglas_is_first_order_in_time():
    """With rho != 0 the explicit mixed term makes Douglas first order: doubling the time
    steps roughly halves its time error, while HV's and CS's drop ~4x."""
    kw = dict(n_S=100, n_v=50)
    for scheme, lo, hi in (("douglas", 1.6, 2.6), ("hv", 3.3, 4.5), ("cs", 3.3, 4.5)):
        fine = heston_adi(100, 120, 1.0, 0.025, IHF, scheme=scheme, n_time=800, **kw)
        err = lambda n: abs(heston_adi(100, 120, 1.0, 0.025, IHF, scheme=scheme, n_time=n, **kw) - fine)
        assert lo < err(20) / err(40) < hi, scheme


def test_adi_put_by_parity():
    ref = cos_price(100, 110, 0.5, 0.025, IHF, call=False)
    assert heston_adi(100, 110, 0.5, 0.025, IHF, call=False, n_S=200, n_v=100) == pytest.approx(ref, abs=5e-3)


def test_adi_american_put_benchmark():
    """Clarke & Parrott (1999) / Ikonen & Toivanen (2009): K = 10, T = 0.25, r = 0.1."""
    p = HestonParams(v0=0.0625, kappa=5.0, theta=0.16, sigma=0.9, rho=0.1)
    ref = {8: 2.0000, 9: 1.107641, 10: 0.520030, 11: 0.213668, 12: 0.082036}
    for S, v in ref.items():
        got = heston_adi(S, 10, 0.25, 0.1, p, call=False, american=True,
                         n_S=200, n_v=100, n_time=200)
        assert got == pytest.approx(v, abs=1e-3)
