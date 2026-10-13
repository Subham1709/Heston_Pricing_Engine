import numpy as np
import pytest

from heston.fourier import cos_price
from heston.model import HestonParams
from heston.monte_carlo import mc_price, simulate

FO = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)


def test_qe_matches_fourier_within_error_bars():
    ref = cos_price(100, 100, 1.0, 0.0, FO)
    price, se = mc_price(100, 100, 1.0, 0.0, FO, n_paths=200_000, n_steps=32, scheme="qe", seed=1)
    assert abs(price - ref) < 4 * se + 2e-3


def test_qe_less_biased_than_euler_at_coarse_steps():
    """Feller condition fails here (0.38), the hard case for Euler."""
    ref = cos_price(100, 100, 1.0, 0.0, FO)
    qe, _ = mc_price(100, 100, 1.0, 0.0, FO, n_paths=200_000, n_steps=4, scheme="qe", seed=2)
    eu, _ = mc_price(100, 100, 1.0, 0.0, FO, n_paths=200_000, n_steps=4, scheme="euler", seed=2)
    assert abs(qe - ref) < 0.2 * abs(eu - ref)


@pytest.mark.parametrize("scheme", ["qe", "euler"])
def test_discounted_stock_is_martingale(scheme):
    ST, vT = simulate(100, 1.0, 0.05, FO, n_paths=200_000, n_steps=50, q=0.02, scheme=scheme, seed=3)
    assert np.mean(ST) * np.exp(-0.05) == pytest.approx(100 * np.exp(-0.02), rel=3e-3)
    assert vT.min() >= 0


def test_variance_mean_reverts():
    _, vT = simulate(100, 5.0, 0.0, FO, n_paths=100_000, n_steps=100, seed=4)
    expected = FO.theta + (FO.v0 - FO.theta) * np.exp(-FO.kappa * 5.0)
    assert np.mean(vT) == pytest.approx(expected, rel=0.02)
