"""Calibrating Heston to a market option chain.

Calibration = find the five parameters (v0, kappa, theta, sigma, rho) whose model prices
best match the observed prices. Steps, as a desk does it:

1. Back out the forward F and discount factor D from put-call parity, C - P = D (F - K),
   using near-the-money strikes (one funding rate from the longer expiries, then a
   forward per expiry). This avoids guessing the dividend yield separately.
2. Keep out-of-the-money options only (puts below F, calls above). They are the liquid
   ones, and in-the-money prices add no information beyond parity.
3. Minimise sum_i ((model_i - market_i) / vega_i)^2 with Levenberg-Marquardt / trust
   region least squares. Dividing a price error by vega turns it into an implied-vol
   error, so a 1-cent miss on a cheap wing option counts as much as it should.

The problem is ill-posed in places: kappa and sigma trade off against each other (both
shape how the smile flattens with maturity), so different starts can land on different
kappa with almost the same fit. We use several starts and report the best.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .black_scholes import bs_greeks, implied_vol
from .fourier import cos_price
from .model import HestonParams

BOUNDS = (np.array([1e-4, 0.05, 1e-4, 0.05, -0.999]),   # v0, kappa, theta, sigma, rho
          np.array([1.0, 15.0, 1.0, 3.0, 0.999]))


def forward_and_discount(chain: pd.DataFrame, S: float, moneyness_band=0.1, min_T_for_rate=0.25):
    """Forward F and discount factor D per expiry from put-call parity, C - P = D (F - K).

    1. For each expiry, regress (C - P) on K near the money: slope = -D, intercept = D F.
    2. The slope is noisy for short expiries (D is ~1 and quotes are a few cents apart),
       so take one funding rate r as the median of the rates implied by expiries of at
       least `min_T_for_rate` years, and set D = e^{-rT} for every expiry.
    3. With D fixed, F = K + (C - P) / D at each near-the-money strike; take the median.

    chain columns: expiry, T, K, call (bool), mid. Returns DataFrame (expiry, T, F, D, r, q).
    """
    pairs = []
    for (exp, T), g in chain.groupby(["expiry", "T"]):
        c = g[g.call].groupby("K").mid.mean()
        p = g[~g.call].groupby("K").mid.mean()
        both = c.index.intersection(p.index)
        both = both[np.abs(both / S - 1) < moneyness_band]
        if len(both) >= 4:
            pairs.append((exp, T, both.values.astype(float), (c[both] - p[both]).values))
    if not pairs:
        return pd.DataFrame(columns=["expiry", "T", "F", "D", "r", "q"])

    rates = []
    for _, T, K, y in pairs:
        D = -np.polyfit(K, y, 1)[0]
        if T >= min_T_for_rate and 0.5 < D <= 1.05:
            rates.append(-np.log(D) / T)
    r = float(np.median(rates)) if rates else 0.0

    rows = []
    for exp, T, K, y in pairs:
        D = np.exp(-r * T)
        F = float(np.median(K + y / D))
        rows.append({"expiry": exp, "T": T, "F": F, "D": D, "r": r, "q": r - np.log(F / S) / T})
    return pd.DataFrame(rows)


def prepare_quotes(chain: pd.DataFrame, S: float, min_T=14 / 365, max_T=2.0,
                   k_range=(0.7, 1.3), max_rel_spread=0.25):
    """Clean a raw chain into OTM quotes with forward, rates and market implied vols."""
    ch = chain.copy()
    ch = ch[(ch.bid > 0) & (ch.ask > ch.bid)]
    ch["mid"] = 0.5 * (ch.bid + ch.ask)
    ch = ch[(ch.ask - ch.bid) / ch.mid < max_rel_spread]
    ch = ch[(ch["T"] >= min_T) & (ch["T"] <= max_T)]
    fd = forward_and_discount(ch, S)
    ch = ch.merge(fd, on=["expiry", "T"])
    m = ch.K / ch.F
    ch = ch[(m > k_range[0]) & (m < k_range[1])]
    ch = ch[(ch.call & (ch.K >= ch.F)) | (~ch.call & (ch.K < ch.F))]  # OTM only
    ch["iv"] = [implied_vol(row.mid, S, row.K, row.T, row.r, row.q, row.call)
                for row in ch.itertuples()]
    ch = ch.dropna(subset=["iv"])
    ch = ch[(ch.iv > 0.01) & (ch.iv < 2.0)]
    ch["vega"] = [bs_greeks(S, row.K, row.T, row.r, row.iv, row.q, row.call)["vega"]
                  for row in ch.itertuples()]
    return ch.reset_index(drop=True)


def model_prices(params: HestonParams, quotes: pd.DataFrame, S: float, N=256):
    """Heston prices for every quote; one vectorised COS call per expiry."""
    out = np.empty(len(quotes))
    for _, g in quotes.groupby("T"):
        r, q = g.r.iloc[0], g.q.iloc[0]
        for is_call in (True, False):
            sub = g[g.call == is_call]
            if len(sub):
                out[sub.index] = cos_price(S, sub.K.values, sub["T"].iloc[0], r, params, q,
                                           call=is_call, N=N)
    return out


def calibrate(quotes: pd.DataFrame, S: float, starts=None, feller_penalty=0.0):
    """Vega-weighted least squares. Returns (best params, result dict)."""
    if starts is None:
        atm = float(np.median(quotes.iv) ** 2)
        starts = [
            [atm, 2.0, atm, 0.5, -0.7],
            [atm, 0.5, 1.5 * atm, 1.0, -0.8],
            [atm, 5.0, atm, 1.5, -0.6],
            [0.5 * atm, 1.0, 2 * atm, 0.3, -0.5],
        ]
    mkt = quotes.mid.values
    w = 1.0 / np.maximum(quotes.vega.values, 1e-3 * S)

    def resid(x):
        p = HestonParams(*x)
        res = (model_prices(p, quotes, S) - mkt) * w
        if feller_penalty:
            res = np.append(res, feller_penalty * max(0.0, 1.0 - p.feller()))
        return res

    best = None
    for x0 in starts:
        x0 = np.clip(x0, BOUNDS[0] + 1e-6, BOUNDS[1] - 1e-6)
        sol = least_squares(resid, x0, bounds=BOUNDS, method="trf", x_scale="jac",
                            xtol=1e-10, ftol=1e-10, max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    p = HestonParams(*map(float, best.x))
    fit = model_prices(p, quotes, S)
    model_iv = np.array([implied_vol(fit[i], S, row.K, row.T, row.r, row.q, row.call)
                         for i, row in enumerate(quotes.itertuples())])
    rmse_iv = float(np.sqrt(np.nanmean((model_iv - quotes.iv.values) ** 2)))
    return p, {"rmse_iv": rmse_iv, "model_iv": model_iv, "model_price": fit,
               "nfev": int(best.nfev), "cost": float(best.cost)}
