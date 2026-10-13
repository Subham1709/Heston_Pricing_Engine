"""Finite differences in two dimensions: the Heston PDE with ADI time stepping.

In time-to-expiry tau the Heston price u(S, v, tau) solves

    u_tau = 0.5 v S^2 u_SS + rho sigma v S u_Sv + 0.5 sigma^2 v u_vv
            + (r - q) S u_S + kappa (theta - v) u_v - r u

Split the right-hand side into three operators (In 't Hout & Foulon, 2010):

    A0 = rho sigma v S d^2/dSdv                              (mixed derivative)
    A1 = 0.5 v S^2 d^2/dS^2 + (r - q) S d/dS - r/2           (S direction)
    A2 = 0.5 sigma^2 v d^2/dv^2 + kappa (theta - v) d/dv - r/2 (v direction)

A fully implicit step would need to solve one big 2D sparse system. ADI (alternating
direction implicit) schemes treat A1 and A2 implicitly one at a time, each a
tridiagonal-like solve along one direction, and the awkward mixed term A0 explicitly.
The three schemes here differ in how they correct the explicit A0 part:

    Douglas (Do)               first order in time when rho != 0
    Craig-Sneyd (CS)           second order, theta = 1/2
    Hundsdorfer-Verwer (HV)    second order, theta = 1/2 + sqrt(3)/6, the most robust

Grids are non-uniform: points are packed around the strike in S and near v = 0, where
the solution bends most, using the sinh maps from In 't Hout & Foulon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import splu

from .model import HestonParams


# ------------------------------------------------------------------ grids and stencils

def sinh_grid(lo, hi, n, centre, c):
    """n+1 points on [lo, hi], dense around `centre` (density set by c, smaller = denser)."""
    xi = np.linspace(np.arcsinh((lo - centre) / c), np.arcsinh((hi - centre) / c), n + 1)
    g = centre + c * np.sinh(xi)
    g[0], g[-1] = lo, hi
    return g


def _diff_matrices(x):
    """First and second derivative matrices on a non-uniform grid (3-point central,
    second-order accurate). The first and last rows are left empty (boundaries)."""
    n = len(x)
    h = np.diff(x)
    D1 = sp.lil_matrix((n, n)); D2 = sp.lil_matrix((n, n))
    for i in range(1, n - 1):
        hm, hp = h[i - 1], h[i]
        D1[i, i - 1] = -hp / (hm * (hm + hp))
        D1[i, i] = (hp - hm) / (hm * hp)
        D1[i, i + 1] = hm / (hp * (hm + hp))
        D2[i, i - 1] = 2 / (hm * (hm + hp))
        D2[i, i] = -2 / (hm * hp)
        D2[i, i + 1] = 2 / (hp * (hm + hp))
    return D1.tocsr(), D2.tocsr()


def _forward_d1_at_zero(x):
    """One-sided second-order first derivative at x[0] (used at v = 0)."""
    h0, h1 = x[1] - x[0], x[2] - x[1]
    return np.array([-(2 * h0 + h1) / (h0 * (h0 + h1)), (h0 + h1) / (h0 * h1), -h0 / (h1 * (h0 + h1))])


@dataclass
class HestonGrid:
    S: np.ndarray
    v: np.ndarray


def make_grid(K, n_S=100, n_v=50, S_max_mult=8.0, v_max=5.0):
    S = sinh_grid(0.0, S_max_mult * K, n_S, K, K / 5)
    v = sinh_grid(0.0, v_max, n_v, 0.0, v_max / 500)
    return HestonGrid(S, v)


# ------------------------------------------------------------------ operators

def build_operators(g: HestonGrid, p: HestonParams, r, q):
    """Sparse A0, A1, A2 on the flattened (S, v) grid, index = i * n_v + j."""
    S, v = g.S, g.v
    nS, nv = len(S), len(v)
    D1S, D2S = _diff_matrices(S)
    D1v, D2v = _diff_matrices(v)
    # at v = 0 the PDE degenerates to u_tau = (r-q) S u_S + kappa theta u_v - r u:
    # use a one-sided derivative in v there
    D1v = D1v.tolil(); D1v[0, :3] = _forward_d1_at_zero(v); D1v = D1v.tocsr()

    IS, Iv = sp.identity(nS, format="csr"), sp.identity(nv, format="csr")
    SS, VV = np.meshgrid(S, v, indexing="ij")
    s_flat, v_flat = SS.ravel(), VV.ravel()
    diag = lambda a: sp.diags(a, format="csr")

    A1 = (diag(0.5 * v_flat * s_flat**2) @ sp.kron(D2S, Iv)
          + diag((r - q) * s_flat) @ sp.kron(D1S, Iv) - 0.5 * r * sp.identity(nS * nv))
    A2 = (diag(0.5 * p.sigma**2 * v_flat) @ sp.kron(IS, D2v)
          + diag(p.kappa * (p.theta - v_flat)) @ sp.kron(IS, D1v) - 0.5 * r * sp.identity(nS * nv))
    A0 = diag(p.rho * p.sigma * v_flat * s_flat) @ sp.kron(D1S, D1v)

    # boundary nodes are set directly (Dirichlet), so zero their rows:
    # S = 0, S = S_max and v = v_max. The v = 0 row stays: the PDE holds there.
    boundary = np.zeros((nS, nv), dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, -1] = True
    keep = diag((~boundary).ravel().astype(float))
    return (keep @ A0).tocsr(), (keep @ A1).tocsr(), (keep @ A2).tocsr(), boundary


def _call_boundary(g: HestonGrid, K, tau, r, q):
    """Call values on the Dirichlet boundaries:
    S = 0 -> 0;  S = S_max -> S e^{-q tau} - K e^{-r tau} (deep in the money);
    v = v_max -> S e^{-q tau} (with huge variance the call is worth the stock)."""
    B = np.zeros((len(g.S), len(g.v)))
    B[-1, :] = g.S[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
    B[:, -1] = g.S * np.exp(-q * tau)
    return B


# ------------------------------------------------------------------ time stepping

SCHEMES = {"douglas": 0.5, "cs": 0.5, "hv": 0.5 + np.sqrt(3) / 6}


def heston_adi(S0, K, T, r, p: HestonParams, q=0.0, call=True, scheme="hv",
               n_S=100, n_v=50, n_time=50, damping_steps=0, american=False,
               return_grid=False):
    """European (or, approximately, American) option on the Heston PDE via ADI.

    damping_steps: replace the first steps by pairs of fully implicit Douglas half-steps
    (theta = 1), the 2D analogue of Rannacher smoothing. It damps the oscillations the
    payoff kink can cause in Gamma, at the price of a first-order error term in the
    price, so it is off by default (HV and CS are stable without it here).

    American puts: after each step, V = max(V, payoff) (explicit projection). Simple and
    first order in time; fine for a demonstration.
    """
    g = make_grid(K, n_S, n_v)
    A0, A1, A2, bnd = build_operators(g, p, r, q)
    A = A0 + A1 + A2
    n = A.shape[0]
    I = sp.identity(n, format="csc")
    SS = np.repeat(g.S[:, None], len(g.v), axis=1)

    if american and call:
        raise ValueError("an American call on a non-dividend stock equals the European one")
    payoff_c = np.maximum(SS - K, 0.0)
    payoff = payoff_c if call or not american else np.maximum(K - SS, 0.0)

    # American puts are priced directly; Europeans are priced as calls (the call's
    # boundaries are simpler) and turned into puts by parity at the end.
    if american:
        V = payoff.ravel().copy()
        boundary_fn = lambda tau: _put_boundary(g, K, tau, r, q)
    else:
        V = payoff_c.ravel().copy()
        boundary_fn = lambda tau: _call_boundary(g, K, tau, r, q)
    b_idx = bnd.ravel()

    lu_cache = {}

    def solve(j, th, dt, rhs):
        key = (j, th, dt)
        if key not in lu_cache:
            Aj = A1 if j == 1 else A2
            lu_cache[key] = splu((I - th * dt * Aj).tocsc())
        return lu_cache[key].solve(rhs)

    def step(V, tau_new, dt, scheme, th):
        Bv = boundary_fn(tau_new).ravel()[b_idx]
        AV = A @ V
        Y0 = V + dt * AV
        Y1 = solve(1, th, dt, Y0 - th * dt * (A1 @ V)); Y1[b_idx] = Bv
        Y2 = solve(2, th, dt, Y1 - th * dt * (A2 @ V)); Y2[b_idx] = Bv
        if scheme == "douglas":
            return Y2
        if scheme == "cs":
            Z0 = Y0 + 0.5 * dt * (A0 @ (Y2 - V))
            Z1 = solve(1, th, dt, Z0 - th * dt * (A1 @ V)); Z1[b_idx] = Bv
            Z2 = solve(2, th, dt, Z1 - th * dt * (A2 @ V)); Z2[b_idx] = Bv
            return Z2
        if scheme == "hv":
            Z0 = Y0 + 0.5 * dt * (A @ (Y2 - V))
            Z1 = solve(1, th, dt, Z0 - th * dt * (A1 @ Y2)); Z1[b_idx] = Bv
            Z2 = solve(2, th, dt, Z1 - th * dt * (A2 @ Y2)); Z2[b_idx] = Bv
            return Z2
        raise ValueError(scheme)

    dt = T / n_time
    tau = 0.0
    th = SCHEMES[scheme]
    for _ in range(2 * damping_steps):
        tau += dt / 2
        V = step(V, tau, dt / 2, "douglas", 1.0)
        if american:
            V = np.maximum(V, payoff.ravel())
    for _ in range(n_time - damping_steps):
        tau += dt
        V = step(V, tau, dt, scheme, th)
        if american:
            V = np.maximum(V, payoff.ravel())

    U = V.reshape(len(g.S), len(g.v))
    spline = RectBivariateSpline(g.S, g.v, U, kx=3, ky=3)
    price = float(spline(S0, p.v0)[0, 0])
    if not call and not american:
        price = price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    if return_grid:
        return price, g, U
    return price


def _put_boundary(g: HestonGrid, K, tau, r, q):
    """American put boundaries: S = 0 -> K (exercise at once); S = S_max -> 0;
    v = v_max -> K (huge variance: worth the most it can be, the strike)."""
    B = np.zeros((len(g.S), len(g.v)))
    B[0, :] = K
    B[:, -1] = K
    B[-1, :] = 0.0
    return B
