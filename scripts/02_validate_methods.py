"""Step 7: every pricing method against every other, plus convergence and speed.

Writes results/validation.md and figures to results/. Runs in a few minutes on a laptop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _style import COLORS, plt  # noqa: E402

from heston.black_scholes import bs_price, implied_vol  # noqa: E402
from heston.fd1d import bs_fd, bs_fd_greeks_on_grid  # noqa: E402
from heston.fd_adi import heston_adi  # noqa: E402
from heston.fourier import carr_madan_fft, cos_price, heston_price_integral  # noqa: E402
from heston.model import HestonParams  # noqa: E402
from heston.monte_carlo import mc_price  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)

FO = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
IHF = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.9)


def timed(f, repeat=3):
    best, out = np.inf, None
    for _ in range(repeat):
        t = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t)
    return out, best


def md(df, floatfmt=".6g"):
    df = df.copy()
    for c in df.columns:  # keep integer columns (grid sizes, steps) as integers
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df.to_markdown(index=False, floatfmt=floatfmt)


lines = ["# Validation: one option, five methods\n"]

# ---------------------------------------------------------------- 1. headline table
S, K, T, r = 100.0, 100.0, 1.0, 0.0
ref = 5.785155450  # Fang & Oosterlee (2008)
rows = []
for name, f, rep in [
    ("Gil-Pelaez integral (adaptive quadrature)", lambda: heston_price_integral(S, K, T, r, FO), 3),
    ("Carr-Madan FFT (N = 4096)", lambda: carr_madan_fft(S, K, T, r, FO), 3),
    ("COS (N = 256)", lambda: cos_price(S, K, T, r, FO), 3),
    ("Monte Carlo, Euler, 100 steps, 400k paths", lambda: mc_price(S, K, T, r, FO, n_paths=400_000, n_steps=100, scheme="euler")[0], 1),
    ("Monte Carlo, Andersen QE, 100 steps, 400k paths", lambda: mc_price(S, K, T, r, FO, n_paths=400_000, n_steps=100, scheme="qe")[0], 1),
    ("PDE, ADI Hundsdorfer-Verwer, 200 x 100 x 100", lambda: heston_adi(S, K, T, r, FO, n_S=200, n_v=100, n_time=100), 1),
    ("PDE, ADI Craig-Sneyd, 200 x 100 x 100", lambda: heston_adi(S, K, T, r, FO, scheme="cs", n_S=200, n_v=100, n_time=100), 1),
]:
    v, t = timed(f, rep)
    rows.append({"Method": name, "Price": v, "Error vs reference": v - ref, "Time (ms)": 1e3 * t})
tab = pd.DataFrame(rows)
lines += ["Fang & Oosterlee (2008) test case: S = K = 100, T = 1, r = 0, v0 = 0.0175, kappa = 1.5768, "
          "theta = 0.0398, sigma = 0.5751, rho = -0.5711 (Feller ratio 0.38, so the variance hits zero). "
          f"Reference price {ref}.\n", md(tab), ""]
tab.to_csv(RES / "validation_headline.csv", index=False)
print(tab)

# ---------------------------------------------------------------- 2. COS convergence
Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256]
cos_err = [abs(cos_price(S, K, T, r, FO, N=n) - ref) for n in Ns]
fft_N = [256, 512, 1024, 2048, 4096, 8192]
fft_err = [abs(carr_madan_fft(S, K, T, r, FO, N=n) - ref) for n in fft_N]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.semilogy(Ns, np.maximum(cos_err, 1e-16), "o-", label="COS")
ax.semilogy(fft_N, np.maximum(fft_err, 1e-16), "s-", label="Carr-Madan FFT")
ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256, 1024, 4096]); ax.set_xticklabels([16, 64, 256, 1024, 4096])
ax.set_xlabel("number of terms / FFT points N"); ax.set_ylabel("absolute price error")
ax.set_title("Fourier methods: COS converges exponentially", loc="left")
ax.legend(); fig.tight_layout(); fig.savefig(RES / "fourier_convergence.png"); plt.close(fig)
lines += ["## Fourier convergence\n", "![](fourier_convergence.png)\n",
          md(pd.DataFrame({"N": Ns, "COS error": cos_err}), ".2e"), ""]

# ---------------------------------------------------------------- 3. MC bias vs steps
steps = [2, 4, 8, 16, 32, 64]
mc_rows = []
for sch in ("euler", "qe"):
    for n in steps:
        v, se = mc_price(S, K, T, r, FO, n_paths=1_000_000, n_steps=n, scheme=sch, seed=7)
        mc_rows.append({"scheme": sch, "steps": n, "bias": v - ref, "std error": se})
mc = pd.DataFrame(mc_rows)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, (sch, lab) in enumerate((("euler", "Euler, full truncation"), ("qe", "Andersen QE"))):
    d = mc[mc.scheme == sch]
    ax.errorbar(d.steps, d.bias, yerr=2 * d["std error"], fmt="o-", capsize=3, label=lab, color=COLORS[i])
ax.axhline(0, color="#52514e", lw=1)
ax.set_xscale("log", base=2); ax.set_xticks(steps); ax.set_xticklabels(steps); ax.set_xlabel("time steps per year"); ax.set_ylabel("price bias (MC - exact)")
ax.set_title("Monte Carlo bias: QE is nearly unbiased even with few steps", loc="left")
ax.legend(); fig.tight_layout(); fig.savefig(RES / "mc_bias.png"); plt.close(fig)
lines += ["## Monte Carlo discretisation bias (1M paths, antithetic + control variate)\n",
          "![](mc_bias.png)\n", md(mc, ".4f"), ""]

# control-variate effect
no_cv = mc_price(S, 90, T, r, FO, n_paths=400_000, n_steps=32, control_variate=False)[1]
cv = mc_price(S, 90, T, r, FO, n_paths=400_000, n_steps=32, control_variate=True)[1]
lines += [f"Control variate (stock price), K = 90 call: standard error {no_cv:.4f} -> {cv:.4f} "
          f"({100 * (1 - (cv / no_cv) ** 2):.0f}% less variance, same paths).\n"]

# ---------------------------------------------------------------- 4. CN vs Rannacher Gamma
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, (lab, kw) in zip(axes, (("Crank-Nicolson, no smoothing", dict(rannacher_steps=0)),
                                ("Crank-Nicolson + Rannacher", dict(rannacher_steps=2)))):
    _, Sg, Vg = bs_fd(100, 100, 0.1, 0.05, 0.2, n_space=400, n_time=10, width=4, return_grid=True, **kw)
    s, _, gam = bs_fd_greeks_on_grid(Sg, Vg)
    m = (s > 85) & (s < 115)
    exact = np.exp(-0.5 * ((np.log(s / 100) + (0.05 + 0.02) * 0.1) / (0.2 * np.sqrt(0.1))) ** 2) / (
        s * 0.2 * np.sqrt(0.1) * np.sqrt(2 * np.pi))
    ax.plot(s[m], exact[m], color="#b5b4ad", lw=4, label="exact")
    ax.plot(s[m], gam[m], color=COLORS[0], label="finite differences")
    ax.set_title(lab, loc="left"); ax.set_xlabel("spot")
axes[0].set_ylabel("Gamma"); axes[0].legend()
fig.suptitle("Short-dated call, 10 large time steps: CN rings around the strike", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(RES / "rannacher_gamma.png"); plt.close(fig)
lines += ["## Crank-Nicolson and Rannacher smoothing\n", "![](rannacher_gamma.png)\n"]

ns = [50, 100, 200, 400, 800]
bs_ref = bs_price(100, 100, 1, 0.05, 0.2)
cn = pd.DataFrame({"grid points": ns,
                   "Crank-Nicolson error": [abs(bs_fd(100, 100, 1, 0.05, 0.2, n_space=n, n_time=n // 2) - bs_ref) for n in ns],
                   "implicit Euler error": [abs(bs_fd(100, 100, 1, 0.05, 0.2, n_space=n, n_time=n // 2, theta=1.0) - bs_ref) for n in ns]})
lines += ["Black-Scholes ATM call (exact %.6f), time steps = grid points / 2:\n" % bs_ref, md(cn, ".2e"), "",
          "Halving the grid divides the Crank-Nicolson error by ~4 (second order) and the implicit "
          "Euler error by ~2 (first order in time).\n"]
am = bs_fd(100, 100, 1, 0.05, 0.2, call=False, american=True, n_space=800, n_time=400)
lines += [f"American put (S = K = 100, r = 5%, vol = 20%, T = 1) by PSOR: {am:.4f} "
          f"vs European {bs_price(100, 100, 1, 0.05, 0.2, call=False):.4f}; literature value 6.0904.\n"]

# ---------------------------------------------------------------- 5. ADI convergence
Kc, Tc, rc = 100.0, 1.0, 0.025
adi_ref = cos_price(100, Kc, Tc, rc, IHF, N=512)
nts = [10, 20, 40, 80, 160]
fine = {s: heston_adi(100, Kc, Tc, rc, IHF, scheme=s, n_S=100, n_v=50, n_time=1280) for s in ("douglas", "cs", "hv")}
adi_t = pd.DataFrame({"time steps": nts, **{
    s: [abs(heston_adi(100, Kc, Tc, rc, IHF, scheme=s, n_S=100, n_v=50, n_time=n) - fine[s]) for n in nts]
    for s in ("douglas", "cs", "hv")}})
grids = [(25, 12), (50, 25), (100, 50), (200, 100), (400, 200)]
adi_s = pd.DataFrame({"grid (S x v)": [f"{a} x {b}" for a, b in grids],
                      "HV error vs COS": [abs(heston_adi(100, Kc, Tc, rc, IHF, n_S=a, n_v=b, n_time=100) - adi_ref) for a, b in grids]})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
for i, (s, lab) in enumerate((("douglas", "Douglas"), ("cs", "Craig-Sneyd"), ("hv", "Hundsdorfer-Verwer"))):
    axes[0].loglog(adi_t["time steps"], adi_t[s], "o-", label=lab, color=COLORS[i])
axes[0].set_xlabel("time steps"); axes[0].set_ylabel("time-discretisation error"); axes[0].legend()
axes[0].set_title("ADI schemes in time", loc="left")
axes[1].loglog([a for a, _ in grids], adi_s["HV error vs COS"], "o-", color=COLORS[0])
axes[1].set_xlabel("grid points in S (v grid = half)"); axes[1].set_ylabel("error vs COS")
axes[1].set_title("Spatial convergence (HV)", loc="left")
fig.tight_layout(); fig.savefig(RES / "adi_convergence.png"); plt.close(fig)
lines += ["## Heston PDE with ADI\n",
          "In 't Hout & Foulon case: K = 100, T = 1, r = 2.5%, kappa = 1.5, theta = 0.04, sigma = 0.3, "
          f"rho = -0.9, v0 = 0.04. COS reference {adi_ref:.6f}.\n", "![](adi_convergence.png)\n",
          "Time-discretisation error (vs the same grid with 1280 steps):\n", md(adi_t, ".2e"), "",
          "Spatial error (100 time steps):\n", md(adi_s, ".2e"), ""]

# ---------------------------------------------------------------- 6. smile intuition
Ks = np.linspace(70, 130, 61)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
base = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
for i, rho in enumerate((-0.7, 0.0, 0.7)):
    p = HestonParams(**{**base, "rho": rho})
    iv = [implied_vol(c, 100, k, 0.5, 0.0) for c, k in zip(cos_price(100, Ks, 0.5, 0.0, p), Ks)]
    axes[0].plot(Ks, iv, label=f"rho = {rho}", color=COLORS[i])
for i, sg in enumerate((0.1, 0.5, 1.0)):
    p = HestonParams(**{**base, "sigma": sg, "rho": 0.0})
    iv = [implied_vol(c, 100, k, 0.5, 0.0) for c, k in zip(cos_price(100, Ks, 0.5, 0.0, p), Ks)]
    axes[1].plot(Ks, iv, label=f"sigma = {sg}", color=COLORS[i])
axes[0].set_title("rho tilts the smile (skew)", loc="left"); axes[1].set_title("vol of vol bends it (curvature)", loc="left")
for ax in axes:
    ax.set_xlabel("strike"); ax.legend()
axes[0].set_ylabel("Black-Scholes implied vol, T = 0.5")
fig.tight_layout(); fig.savefig(RES / "smile_parameters.png"); plt.close(fig)

(RES / "validation.md").write_text("\n".join(lines))
print("wrote results/validation.md")
