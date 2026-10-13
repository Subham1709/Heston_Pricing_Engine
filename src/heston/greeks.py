"""Greeks under Heston, three ways.

    cos_greeks       bump-and-reprice with the COS pricer (central differences). Exact up to
                     O(h^2), and cheap because COS is fast. What a desk risk system does.
    adi_greeks       Delta and Gamma read off the PDE grid: one solve gives them at every spot.
    mc_delta_pathwise  pathwise estimator: d/dS0 of the discounted payoff, averaged over paths.
                     For a call that is e^{-rT} 1{S_T > K} S_T / S0, since S_T is linear in S0.

Minimum-variance delta: delta holds v fixed, but in Heston v moves with S
(correlation rho). Regressing dC on dS gives  delta + rho sigma / S * dC/dv,  the hedge
ratio that minimises the variance of the hedged position (Bakshi, Cao & Chen 1997).

"Vega" in Heston has two parts: sensitivity to the current variance v0 and to the long-run
level theta. We report dV/d(sqrt v0), the closest analogue to Black-Scholes vega.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .fd_adi import heston_adi
from .fourier import cos_price
from .model import HestonParams
from .monte_carlo import simulate


def cos_greeks(S, K, T, r, p: HestonParams, q=0.0, call=True, h=1e-3):
    f = lambda S_=S, p_=p: cos_price(S_, K, T, r, p_, q, call)
    dS = h * S
    up, mid, dn = f(S + dS), f(), f(S - dS)
    vol = np.sqrt(p.v0)
    dv = h * vol
    vega = (f(p_=replace(p, v0=(vol + dv) ** 2)) - f(p_=replace(p, v0=(vol - dv) ** 2))) / (2 * dv)
    dvar = h * p.v0
    dC_dv = (f(p_=replace(p, v0=p.v0 + dvar)) - f(p_=replace(p, v0=p.v0 - dvar))) / (2 * dvar)
    delta = (up - dn) / (2 * dS)
    return {"price": mid, "delta": delta, "gamma": (up - 2 * mid + dn) / dS**2,
            "vega_sqrt_v0": vega, "dC_dv0": dC_dv,
            "mv_delta": delta + p.rho * p.sigma / S * dC_dv}


def adi_greeks(S, K, T, r, p: HestonParams, q=0.0, **kw):
    """Call delta and gamma at (S, v0) from a single PDE solve (spline derivatives)."""
    from scipy.interpolate import RectBivariateSpline

    _, g, U = heston_adi(S, K, T, r, p, q, call=True, return_grid=True, **kw)
    sp = RectBivariateSpline(g.S, g.v, U, kx=3, ky=3)
    return {"delta": float(sp(S, p.v0, dx=1)[0, 0]), "gamma": float(sp(S, p.v0, dx=2)[0, 0])}


def mc_delta_pathwise(S, K, T, r, p: HestonParams, q=0.0, n_paths=400_000, n_steps=64, seed=0):
    ST, _ = simulate(S, T, r, p, n_paths, n_steps, q, "qe", seed)
    est = np.exp(-r * T) * (ST > K) * ST / S
    n = len(est) // 2
    pairs = 0.5 * (est[:n] + est[n:2 * n])
    return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(n))
