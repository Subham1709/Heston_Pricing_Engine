"""Step 8: calibrate Heston to the option chain snapshot(s) in data/.

For each market available (SPX always, NIFTY when the NSE download worked):
clean the chain, back out forwards and rates by put-call parity, calibrate by
vega-weighted least squares on the COS pricer, and save the fit.

Writes results/calibration.md, results/calibration_<market>.csv and smile-fit figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _style import COLORS, plt  # noqa: E402

from heston.calibration import calibrate, prepare_quotes  # noqa: E402

DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)


def plot_fit(q, market, path, max_panels=6):
    exps = sorted(q["T"].unique())
    if len(exps) > max_panels:  # spread the panels across the term structure
        exps = [exps[i] for i in np.linspace(0, len(exps) - 1, max_panels).round().astype(int)]
    cols = 3
    rows = int(np.ceil(len(exps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.0 * rows), squeeze=False)
    for ax, T in zip(axes.ravel(), exps):
        g = q[q["T"] == T].sort_values("K")
        ax.plot(g.K / g.F, g.iv, "o", ms=4, color=COLORS[0], label="market (mid)")
        ax.plot(g.K / g.F, g.model_iv, "-", color=COLORS[1], label="Heston")
        ax.set_title(f"T = {T * 365:.0f} days", loc="left")
        ax.set_xlabel("K / F")
    for ax in axes.ravel()[len(exps):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("implied vol"); axes[0, 0].legend()
    fig.suptitle(f"{market}: Heston fit to the implied vol smile", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    meta = json.loads((DATA / "snapshot.json").read_text())
    out = ["# Calibration to market option prices\n",
           f"Snapshot downloaded {meta['downloaded_utc']} (UTC).\n"]
    for market, fname, k_range in (("SPX", "spx_chain.csv", (0.75, 1.2)),
                                   ("NIFTY", "nifty_chain.csv", (0.85, 1.15))):
        f = DATA / fname
        if not f.exists():
            out += [f"## {market}\n", f"No data: {meta.get(market.lower(), {}).get('error', 'not downloaded')}\n"]
            continue
        S = meta[market.lower()]["spot"]
        chain = pd.read_csv(f)
        q = prepare_quotes(chain, S, k_range=k_range)
        t0 = time.perf_counter()
        p, info = calibrate(q, S)
        secs = time.perf_counter() - t0
        q["model_iv"] = info["model_iv"]
        q["model_price"] = info["model_price"]
        q.to_csv(RES / f"calibration_{market.lower()}.csv", index=False)
        plot_fit(q, market, RES / f"calibration_{market.lower()}.png")

        by_T = (q.assign(err=(q.model_iv - q.iv) * 100).groupby("T")
                .agg(days=("T", lambda t: round(t.iloc[0] * 365)), quotes=("K", "size"),
                     rmse_vol_pts=("err", lambda e: np.sqrt(np.mean(e**2))),
                     r=("r", "first"), q=("q", "first")).reset_index(drop=True))
        out += [f"## {market}\n",
                f"Spot {S:.2f}. {len(q)} out-of-the-money quotes over {q['T'].nunique()} expiries "
                f"({q['T'].min() * 365:.0f} to {q['T'].max() * 365:.0f} days), "
                f"K/F in {k_range}. Calibrated in {secs:.1f} s.\n",
                "| parameter | value |", "|---|---|",
                f"| v0 (vol {np.sqrt(p.v0):.1%}) | {p.v0:.4f} |",
                f"| kappa (half-life {np.log(2) / p.kappa * 12:.1f} months) | {p.kappa:.3f} |",
                f"| theta (vol {np.sqrt(p.theta):.1%}) | {p.theta:.4f} |",
                f"| sigma | {p.sigma:.3f} |", f"| rho | {p.rho:.3f} |",
                f"| Feller ratio 2 kappa theta / sigma^2 | {p.feller():.2f} |", "",
                f"RMSE in implied vol: **{info['rmse_iv'] * 100:.2f} vol points** overall.\n",
                f"![](calibration_{market.lower()}.png)\n",
                "Fit by expiry (rates and dividend yields implied from put-call parity):\n",
                by_T.astype({"days": int, "quotes": int}).to_markdown(index=False, floatfmt=".3f"), ""]
        print(market, p, info["rmse_iv"])
    (RES / "calibration.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
