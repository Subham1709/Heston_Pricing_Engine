"""Heston Pricing Engine: interactive web app.

    streamlit run app/streamlit_app.py

Everything is computed live with the engine in src/heston (no stored prices), except the
SPX calibration, which reads the fit saved by scripts/03_calibrate.py (it takes ~100 s).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heston.black_scholes import bs_greeks, bs_price, implied_vol  # noqa: E402
from heston.calibration import model_prices  # noqa: E402
from heston.fd_adi import heston_adi  # noqa: E402
from heston.fourier import carr_madan_fft, cos_price, heston_price_integral  # noqa: E402
from heston.greeks import cos_greeks  # noqa: E402
from heston.model import HestonParams  # noqa: E402
from heston.monte_carlo import mc_price  # noqa: E402

REPO = "https://github.com/Subham1709/Heston_Pricing_Engine"
SPX = HestonParams(v0=0.0136, kappa=7.408, theta=0.0421, sigma=1.895, rho=-0.648)
C = ["#0f6b5c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]

st.set_page_config(page_title="Heston Pricing Engine", page_icon="📈", layout="wide")


def smile(S, Ks, T, r, p, q=0.0):
    """Implied vol of Heston prices across strikes (OTM options, COS)."""
    out = []
    for call in (False, True):
        sub = Ks[Ks >= S] if call else Ks[Ks < S]
        if len(sub):
            prices = cos_price(S, sub, T, r, p, q, call=call)
            out += [implied_vol(pr, S, k, T, r, q, call) for pr, k in zip(prices, sub)]
    return np.array(out, float)


@st.cache_data
def load_calibration():
    return pd.read_csv(ROOT / "results" / "calibration_spx.csv")


st.title("Heston Pricing Engine")
st.markdown(
    "A from-scratch engine for the **Heston stochastic volatility model**: Fourier (COS, "
    "Carr-Madan FFT, Gil-Pelaez), Monte Carlo (Andersen QE) and finite differences (ADI), "
    "each checked against the others, then **calibrated to 1,880 real SPX option quotes**. "
    f"Code: [GitHub]({REPO}).")

with st.sidebar:
    st.header("Model parameters")
    def reset():
        for k, v in zip(("v0", "kappa", "theta", "sigma", "rho"), SPX.as_array()):
            st.session_state[k] = float(v)

    if "v0" not in st.session_state:
        reset()
    st.button("Reset to SPX calibration (12 Oct 2026)", on_click=reset)
    v0 = st.slider("v0: current variance", 0.005, 0.20, step=0.001, key="v0",
                   format="%.3f", help="sqrt(v0) is today's volatility")
    kappa = st.slider("kappa: speed of mean reversion", 0.1, 10.0, step=0.1,
                      key="kappa")
    theta = st.slider("theta: long-run variance", 0.005, 0.20, step=0.001,
                      key="theta", format="%.3f")
    sigma = st.slider("sigma: volatility of volatility", 0.05, 2.5, step=0.05,
                      key="sigma")
    rho = st.slider("rho: spot-vol correlation", -0.95, 0.5, step=0.01, key="rho")
    p = HestonParams(v0, kappa, theta, sigma, rho)
    st.caption(f"Today's vol {np.sqrt(v0):.1%}, long-run vol {np.sqrt(theta):.1%}, "
               f"Feller ratio 2κθ/σ² = {p.feller():.2f} "
               f"({'variance can hit zero' if p.feller() < 1 else 'variance stays positive'}).")

tab_price, tab_smile, tab_cal, tab_greeks, tab_val = st.tabs(
    ["Price an option", "Smile", "SPX calibration", "Hedge ratios", "Validation"])

# ------------------------------------------------------------------------------ price
with tab_price:
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot S", 1.0, 100_000.0, 100.0)
    K = c2.number_input("Strike K", 1.0, 100_000.0, 100.0)
    T = c3.number_input("Years to expiry T", 0.02, 5.0, 0.5, 0.05)
    r = c4.number_input("Rate r", -0.02, 0.15, 0.03, 0.005, format="%.3f")
    call = c5.radio("Type", ["Call", "Put"], horizontal=True) == "Call"

    rows = []
    for name, fn in (("COS (N = 256)", lambda: cos_price(S, K, T, r, p, call=call)),
                     ("Carr-Madan FFT", lambda: carr_madan_fft(S, K, T, r, p, call=call)),
                     ("Gil-Pelaez integral", lambda: heston_price_integral(S, K, T, r, p, call=call))):
        t0 = time.perf_counter()
        val = float(np.ravel(fn())[0])
        rows.append({"method": name, "price": val, "time (ms)": 1e3 * (time.perf_counter() - t0)})
    ref = rows[0]["price"]
    iv = implied_vol(ref, S, K, T, r, 0.0, call)
    m1, m2, m3 = st.columns(3)
    m1.metric("Heston price (COS)", f"{ref:.4f}")
    m2.metric("Black-Scholes implied vol", f"{iv:.2%}" if np.isfinite(iv) else "n/a")
    m3.metric("Black-Scholes at today's vol", f"{bs_price(S, K, T, r, np.sqrt(v0), call=call):.4f}")

    if st.checkbox("Also run the slow methods (Monte Carlo and the PDE, a few seconds)"):
        t0 = time.perf_counter()
        mc, se = mc_price(S, K, T, r, p, call=call, n_paths=100_000, n_steps=50)
        rows.append({"method": "Monte Carlo, Andersen QE, 100k paths", "price": mc,
                     "time (ms)": 1e3 * (time.perf_counter() - t0), "note": f"± {se:.4f} (1 s.e.)"})
        t0 = time.perf_counter()
        pde = heston_adi(S, K, T, r, p, call=call, n_S=100, n_v=50, n_time=50)
        rows.append({"method": "PDE, ADI Hundsdorfer-Verwer, 100 x 50 x 50",
                     "price": float(np.ravel(pde)[0]),
                     "time (ms)": 1e3 * (time.perf_counter() - t0)})
    df = pd.DataFrame(rows)
    df["difference vs COS"] = df["price"] - ref
    st.dataframe(df.style.format({"price": "{:.6f}", "time (ms)": "{:.1f}",
                                  "difference vs COS": "{:+.1e}"}), width="stretch",
                 hide_index=True)
    st.caption("Same option, independent methods. The Fourier methods agree to ~1e-7; Monte "
               "Carlo agrees within its standard error; the PDE within its grid error.")

# ------------------------------------------------------------------------------ smile
with tab_smile:
    st.markdown(
        "Black-Scholes assumes one volatility for every strike, so its smile is flat. "
        "Heston's random volatility bends it: **rho** tilts it (negative rho = expensive "
        "downside puts, the equity skew), **sigma** curves it, **v0** and **theta** set its "
        "level now and later, **kappa** sets how fast it flattens with maturity. Move the "
        "sliders on the left.")
    S0 = 100.0
    Ks = np.linspace(70, 130, 61)
    fig = go.Figure()
    for T_, col in zip((1 / 12, 0.25, 1.0, 2.0), C):
        ivs = smile(S0, Ks, T_, 0.0, p)
        fig.add_trace(go.Scatter(x=Ks / S0, y=100 * ivs, name=f"T = {T_ * 12:.0f} months",
                                 line=dict(color=col, width=2.5)))
    fig.add_hline(y=100 * np.sqrt(v0), line_dash="dot", line_color="gray",
                  annotation_text="Black-Scholes at today's vol (flat)")
    fig.update_layout(xaxis_title="strike / spot", yaxis_title="implied vol (%)", height=480,
                      title="Implied volatility smile produced by these parameters")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------- calibration
with tab_cal:
    q = load_calibration()
    st.markdown(
        f"SPX option chain, closing quotes of 12 Oct 2026 (spot 7,807): **{len(q):,} "
        f"out-of-the-money quotes over {q['T'].nunique()} expiries**. Forwards, rates and "
        "dividend yields come from put-call parity. Five parameters were fitted to the whole "
        "surface by vega-weighted least squares on the COS pricer: **RMSE 0.90 vol points**.")
    cols = st.columns(6)
    for col, (lbl, val) in zip(cols, [("v0", SPX.v0), ("kappa", SPX.kappa), ("theta", SPX.theta),
                                     ("sigma", SPX.sigma), ("rho", SPX.rho),
                                     ("Feller ratio", SPX.feller())]):
        col.metric(lbl, f"{val:.4g}")
    days = sorted((q["T"] * 365).round().astype(int).unique())
    d = st.select_slider("Expiry (days)", days, value=days[6])
    g = q[(q["T"] * 365).round().astype(int) == d].sort_values("K")
    use_mine = st.toggle("Also show the smile from the sidebar parameters")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.K / g.F, y=100 * g.iv, mode="markers", name="market (mid)",
                             marker=dict(color=C[2], size=6)))
    fig.add_trace(go.Scatter(x=g.K / g.F, y=100 * g.model_iv, name="Heston, calibrated",
                             line=dict(color=C[1], width=3)))
    if use_mine:
        S_spx = 7807.24
        gi = g.reset_index(drop=True)
        mine = model_prices(p, gi, S_spx)
        mine_iv = [implied_vol(mine[i], S_spx, row.K, row.T, row.r, row.q, row.call)
                   for i, row in enumerate(gi.itertuples())]
        fig.add_trace(go.Scatter(x=gi.K / gi.F, y=100 * np.array(mine_iv, float),
                                 name="Heston, your sidebar parameters",
                                 line=dict(color=C[0], dash="dash")))
    rmse = np.sqrt(np.mean((g.model_iv - g.iv) ** 2)) * 100
    fig.update_layout(title=f"{d}-day SPX smile: fit error {rmse:.2f} vol points",
                      xaxis_title="strike / forward", yaxis_title="implied vol (%)", height=480)
    st.plotly_chart(fig, width="stretch")
    by = (q.assign(days=(q["T"] * 365).round().astype(int), err=(q.model_iv - q.iv) * 100)
          .groupby("days").agg(quotes=("K", "size"),
                               rmse_vol_pts=("err", lambda e: np.sqrt(np.mean(e**2)))))
    fig = go.Figure(go.Bar(x=by.index.astype(str), y=by.rmse_vol_pts, marker_color=C[0]))
    fig.update_layout(title="Fit error by expiry", xaxis_title="days to expiry",
                      yaxis_title="RMSE (vol points)", height=320)
    st.plotly_chart(fig, width="stretch")
    st.info("The fit is good from 3 months out (under 0.5 vol points) but Heston cannot make "
            "the steep skew of short-dated options (1.8 vol points at 16 days). That is the "
            "model's known weakness, and why desks add jumps (Bates) or use rough or "
            "local-stochastic volatility.")

# ----------------------------------------------------------------------------- greeks
with tab_greeks:
    st.markdown(
        "Three different numbers for 'the' delta of the same call:\n"
        "- **Black-Scholes delta at the option's implied vol**: assumes implied vol stays put "
        "when spot moves.\n"
        "- **Heston delta**: moves spot, holds the variance v fixed.\n"
        "- **Minimum-variance delta** = Heston delta + (rho sigma / S) dC/dv: also hedges the "
        "part of a vol move that comes with a spot move. With rho < 0 it holds less stock.")
    T_g = st.slider("Years to expiry", 0.05, 2.0, 0.5, 0.05, key="Tg")
    Ks = np.linspace(80, 120, 21)
    rows = []
    for k in Ks:
        gk = cos_greeks(100.0, k, T_g, 0.03, p)
        price = float(np.ravel(gk["price"])[0])
        ivk = implied_vol(price, 100.0, k, T_g, 0.03)
        rows.append({"K": k, "Heston delta": float(np.ravel(gk["delta"])[0]),
                     "min-variance delta": float(np.ravel(gk["mv_delta"])[0]),
                     "BS delta at implied vol": bs_greeks(100.0, k, T_g, 0.03, ivk)["delta"]})
    gd = pd.DataFrame(rows)
    fig = go.Figure()
    for col, c in zip(["BS delta at implied vol", "Heston delta", "min-variance delta"], C[2:] + C[:1]):
        fig.add_trace(go.Scatter(x=gd.K, y=gd[col], name=col, line=dict(width=2.5, color=c)))
    fig.update_layout(xaxis_title="strike (spot = 100)", yaxis_title="delta of a call",
                      height=460, title="Hedge ratios under the sidebar parameters")
    st.plotly_chart(fig, width="stretch")
    atm = gd.iloc[(gd.K - 100).abs().argmin()]
    st.caption(f"At the money: BS {atm['BS delta at implied vol']:.3f}, Heston "
               f"{atm['Heston delta']:.3f}, min-variance {atm['min-variance delta']:.3f}. "
               "Which one hedges best on real paths is what the Deep Hedging project tests.")

# -------------------------------------------------------------------------- validation
with tab_val:
    st.markdown("Fang & Oosterlee (2008) test case, Feller condition violated (the hard case). "
                "Reference price 5.785155450.")
    st.markdown("""
| Method | Error | Time |
|---|---|---|
| Gil-Pelaez integral (adaptive quadrature) | 2e-8 | 1.7 ms |
| Carr-Madan FFT, N = 4096 | 2e-7 | 0.8 ms |
| **COS, N = 256** | **8e-8** | **0.15 ms** |
| Monte Carlo, Euler, 100 steps, 400k paths | 1.5e-2 | 1.4 s |
| Monte Carlo, Andersen QE, 100 steps, 400k paths | 2e-3 (std err 8e-3) | 3.2 s |
| PDE, ADI Hundsdorfer-Verwer, 200 x 100 grid, 100 steps | 6e-4 | 0.26 s |
| PDE, ADI Craig-Sneyd, same grid | 6e-4 | 0.30 s |
""")
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "results" / "fourier_convergence.png"),
             caption="COS converges exponentially in the number of terms", width="stretch")
    c2.image(str(ROOT / "results" / "mc_bias.png"),
             caption="Andersen QE removes the bias Euler has when variance hits zero",
             width="stretch")
    c1.image(str(ROOT / "results" / "adi_convergence.png"),
             caption="ADI schemes: measured convergence orders match theory", width="stretch")
    c2.image(str(ROOT / "results" / "rannacher_gamma.png"),
             caption="Rannacher smoothing removes Crank-Nicolson's Gamma oscillations",
             width="stretch")
    st.markdown(f"Full tables: [results/validation.md]({REPO}/blob/main/results/validation.md). "
                "American puts by PSOR: Black-Scholes 6.0900 vs literature 6.0904; Heston "
                "matches the Clarke-Parrott benchmark to ~4e-4.")
