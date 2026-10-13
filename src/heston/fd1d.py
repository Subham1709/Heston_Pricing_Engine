"""Finite differences in one dimension: the Black-Scholes PDE.

Written in time-to-expiry tau = T - t and log-spot x = log S, the Black-Scholes PDE has
constant coefficients:

    V_tau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V

Discretise x on a uniform grid and use central differences: the right-hand side becomes
a tridiagonal matrix A acting on the vector of option values. The theta-scheme

    (I - theta dt A) V^{n+1} = (I + (1 - theta) dt A) V^n

is explicit Euler for theta = 0 (only stable if dt <= dx^2 / sigma^2), implicit Euler for
theta = 1 (always stable, first order in time), and Crank-Nicolson for theta = 1/2
(second order, but it does not damp high frequencies, so the kink in the payoff makes
spurious oscillations in Gamma). Rannacher smoothing fixes that: start with a few
implicit Euler half-steps, then switch to Crank-Nicolson.

American options add the constraint V >= payoff. Each time step becomes a linear
complementarity problem, solved here with projected SOR (PSOR).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def _payoff(S, K, call):
    return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)


def _boundaries(S_lo, S_hi, K, tau, r, q, call, american):
    """Values at the two ends of the grid (far in and far out of the money)."""
    if call:
        lo = 0.0
        hi = S_hi * np.exp(-q * tau) - K * np.exp(-r * tau)
        if american:
            hi = max(hi, S_hi - K)
    else:
        lo = K * np.exp(-r * tau) - S_lo * np.exp(-q * tau)
        if american:
            lo = max(lo, K - S_lo)
        hi = 0.0
    return lo, hi


def _psor(ab, rhs, payoff, x0, omega=1.5, tol=1e-10, max_iter=10_000):
    """Projected SOR for  A V >= rhs,  V >= payoff,  (A V - rhs)(V - payoff) = 0,
    with A tridiagonal stored in banded form `ab` (upper, diag, lower)."""
    up, dg, lo = ab[0], ab[1], ab[2]
    V = x0.copy()
    n = len(V)
    for _ in range(max_iter):
        err = 0.0
        for i in range(n):
            s = rhs[i]
            if i > 0:
                s -= lo[i - 1] * V[i - 1]
            if i < n - 1:
                s -= up[i + 1] * V[i + 1]
            new = max(payoff[i], V[i] + omega * (s / dg[i] - V[i]))
            err = max(err, abs(new - V[i]))
            V[i] = new
        if err < tol:
            break
    return V


def bs_fd(S0, K, T, r, sigma, q=0.0, call=True, american=False,
          n_space=400, n_time=200, theta=0.5, rannacher_steps=2, width=6.0,
          return_grid=False):
    """Price a European or American option on the Black-Scholes PDE.

    The grid spans log(S0) +/- width * sigma * sqrt(T) and has the strike on a node.
    Returns the price at S0 (and the grid if asked, for plotting Gamma).
    """
    half = width * sigma * np.sqrt(T)
    x_lo, x_hi = np.log(S0) - half, np.log(S0) + half
    dx = (x_hi - x_lo) / n_space
    # shift the grid so log K sits exactly on a node: the payoff kink is then resolved
    x_lo = np.log(K) - dx * np.round((np.log(K) - x_lo) / dx)
    x = x_lo + dx * np.arange(n_space + 1)
    S = np.exp(x)
    dt = T / n_time

    a = 0.5 * sigma**2 / dx**2
    b = (r - q - 0.5 * sigma**2) / (2 * dx)
    # A V_i = lo V_{i-1} + dg V_i + up V_{i+1} on interior nodes
    lo_c, dg_c, up_c = a - b, -2 * a - r, a + b
    m = n_space - 1  # interior unknowns

    def step(Vn, tau_new, th, h):
        # right-hand side (I + (1-th) h A) V^n on interior nodes
        rhs = Vn[1:-1] + (1 - th) * h * (lo_c * Vn[:-2] + dg_c * Vn[1:-1] + up_c * Vn[2:])
        bl, bh = _boundaries(S[0], S[-1], K, tau_new, r, q, call, american)
        rhs[0] += th * h * lo_c * bl
        rhs[-1] += th * h * up_c * bh
        ab = np.zeros((3, m))
        ab[0, 1:] = -th * h * up_c
        ab[1, :] = 1 - th * h * dg_c
        ab[2, :-1] = -th * h * lo_c
        if american:
            pay = _payoff(S[1:-1], K, call)
            inner = _psor(ab, rhs, pay, np.maximum(solve_banded((1, 1), ab, rhs), pay))
        else:
            inner = solve_banded((1, 1), ab, rhs)
        return np.concatenate([[bl], inner, [bh]])

    V = _payoff(S, K, call)
    tau = 0.0
    # Rannacher start-up: 2 * rannacher_steps implicit Euler steps of size dt/2
    for _ in range(2 * rannacher_steps if theta != 1 else 0):
        tau += dt / 2
        V = step(V, tau, 1.0, dt / 2)
    n_cn = n_time - (rannacher_steps if theta != 1 else 0)
    for _ in range(n_cn):
        tau += dt
        V = step(V, tau, theta, dt)

    price = float(np.interp(np.log(S0), x, V))
    if return_grid:
        return price, S, V
    return price


def bs_fd_greeks_on_grid(S, V):
    """Delta and Gamma from the grid by central differences in S (non-uniform in S)."""
    dS_f, dS_b = S[2:] - S[1:-1], S[1:-1] - S[:-2]
    delta = (V[2:] - V[:-2]) / (S[2:] - S[:-2])
    gamma = 2 * ((V[2:] - V[1:-1]) / dS_f - (V[1:-1] - V[:-2]) / dS_b) / (dS_f + dS_b)
    return S[1:-1], delta, gamma
