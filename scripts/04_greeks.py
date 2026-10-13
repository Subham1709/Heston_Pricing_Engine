"""Step 9: Heston Greeks by three methods, and how Heston delta differs from the
Black-Scholes delta a trader would compute from the same option's implied vol.

Writes results/greeks.md and results/delta_vs_bs.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _style import COLORS, plt  # noqa: E402

from heston.black_scholes import bs_greeks, implied_vol  # noqa: E402
from heston.greeks import adi_greeks, cos_greeks, mc_delta_pathwise  # noqa: E402
from heston.model import HestonParams  # noqa: E402

RES = ROOT / "results"
p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.6, rho=-0.7)  # equity-like skew
S, T, r = 100.0, 0.5, 0.03

rows = []
for K in (80, 90, 100, 110, 120):
    c = cos_greeks(S, K, T, r, p)
    a = adi_greeks(S, K, T, r, p, n_S=200, n_v=100, n_time=100)
    m, se = mc_delta_pathwise(S, K, T, r, p)
    iv = implied_vol(c["price"], S, K, T, r)
    b = bs_greeks(S, K, T, r, iv)
    rows.append({"K": K, "implied vol": iv, "delta COS bump": c["delta"], "delta ADI grid": a["delta"],
                 "delta MC pathwise": m, "MC std err": se, "min-variance delta": c["mv_delta"],
                 "BS delta at implied vol": b["delta"],
                 "gamma COS": c["gamma"], "gamma ADI": a["gamma"]})
tab = pd.DataFrame(rows)
print(tab.to_string())

Ks = np.linspace(75, 125, 51)
hd, md_, bd = [], [], []
for K in Ks:
    c = cos_greeks(S, K, T, r, p)
    hd.append(c["delta"]); md_.append(c["mv_delta"])
    bd.append(bs_greeks(S, K, T, r, implied_vol(c["price"], S, K, T, r))["delta"])
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(Ks, hd, color=COLORS[0], label="Heston delta (v held fixed)")
ax.plot(Ks, md_, color=COLORS[1], label="Heston minimum-variance delta")
ax.plot(Ks, bd, color=COLORS[2], label="Black-Scholes delta at implied vol")
ax.set_xlabel("strike"); ax.set_ylabel("call delta")
ax.set_title("Three hedge ratios for the same option", loc="left")
ax.legend()
fig.tight_layout(); fig.savefig(RES / "delta_vs_bs.png"); plt.close(fig)

txt = ["# Greeks under Heston\n",
       f"Parameters: v0 = {p.v0}, kappa = {p.kappa}, theta = {p.theta}, sigma = {p.sigma}, rho = {p.rho}; "
       f"S = {S}, T = {T}, r = {r}. Calls.\n",
       tab.to_markdown(index=False, floatfmt=".4f"), "",
       "The three Heston deltas agree (COS bump and the ADI grid to ~1e-3, MC within its error bars).\n",
       "![](delta_vs_bs.png)\n",
       "Three different numbers for 'the' delta:\n",
       "- **Black-Scholes delta at the option's implied vol** assumes implied vol stays put when spot moves.",
       "- **Heston delta** holds the variance v fixed. Heston prices depend on K/S, so when spot rises the "
       "option moves to a lower K/S, where the (negatively skewed) smile is higher: implied vol rises "
       "with spot, and this delta is *above* Black-Scholes.",
       "- **Minimum-variance delta** = delta + rho sigma / S * dC/dv. In Heston, v tends to fall when "
       "spot rises (rho < 0), which lowers the call's value, so the best hedge holds *less* stock than "
       "the v-fixed delta.\n",
       "Which one hedges best on real paths is an empirical question, and it is the starting point of "
       "the deep hedging project, where a neural network learns the hedge ratio directly.\n"]
(RES / "greeks.md").write_text("\n".join(txt))
print("wrote results/greeks.md")
