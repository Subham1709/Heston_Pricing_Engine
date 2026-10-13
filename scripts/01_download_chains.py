"""Download today's option chains and save them to data/ (runs on a GitHub runner).

- SPX index options from Yahoo Finance (European exercise, the most liquid options market).
- NIFTY index options from NSE's option-chain API (European too). NSE often blocks
  cloud IP ranges, so this part is best-effort: if it fails, only SPX is saved.

Output: data/spx_chain.csv, data/nifty_chain.csv (if available), data/snapshot.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)


def spx_chain(max_expiries=14):
    import yfinance as yf

    tk = yf.Ticker("^SPX")
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    rows = []
    expiries = [e for e in tk.options
                if 10 <= (pd.Timestamp(e) - now).days <= 800]
    # spread the expiries out: all short ones are weeklies, keep every k-th
    step = max(1, len(expiries) // max_expiries)
    for e in expiries[::step][:max_expiries]:
        oc = tk.option_chain(e)
        for df, is_call in ((oc.calls, True), (oc.puts, False)):
            d = df[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
            d["call"] = is_call
            d["expiry"] = e
            rows.append(d)
    ch = pd.concat(rows, ignore_index=True).rename(columns={"strike": "K"})
    # SPX PM-settled options expire at 16:00 New York time
    expiry_ts = pd.to_datetime(ch.expiry) + pd.Timedelta(hours=16)
    ch["T"] = (expiry_ts - now).dt.total_seconds() / (365.0 * 86400)
    return spot, ch


def nifty_chain():
    import requests

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9", "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/option-chain"})
    s.get("https://www.nseindia.com/option-chain", timeout=20)  # sets cookies
    js = s.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", timeout=20).json()
    rec = js["records"]
    spot = float(rec["underlyingValue"])
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    rows = []
    for item in rec["data"]:
        for side, is_call in (("CE", True), ("PE", False)):
            if side in item:
                o = item[side]
                rows.append({"K": float(item["strikePrice"]), "expiry": item["expiryDate"],
                             "bid": o.get("bidprice", 0.0), "ask": o.get("askPrice", 0.0),
                             "lastPrice": o.get("lastPrice", 0.0), "volume": o.get("totalTradedVolume", 0),
                             "openInterest": o.get("openInterest", 0), "call": is_call})
    ch = pd.DataFrame(rows)
    # NIFTY options expire at 15:30 IST
    expiry_ts = pd.to_datetime(ch.expiry, format="%d-%b-%Y") + pd.Timedelta(hours=15, minutes=30)
    ch["T"] = (expiry_ts - now).dt.total_seconds() / (365.0 * 86400)
    ch["expiry"] = expiry_ts.dt.strftime("%Y-%m-%d")
    return spot, ch


def main():
    meta = {"downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    spot, ch = spx_chain()
    ch.to_csv(DATA / "spx_chain.csv", index=False)
    meta["spx"] = {"spot": spot, "rows": len(ch), "expiries": int(ch.expiry.nunique())}
    print("SPX", meta["spx"])
    try:
        spot, ch = nifty_chain()
        ch.to_csv(DATA / "nifty_chain.csv", index=False)
        meta["nifty"] = {"spot": spot, "rows": len(ch), "expiries": int(ch.expiry.nunique())}
        print("NIFTY", meta["nifty"])
    except Exception as exc:  # NSE blocks many cloud IPs
        meta["nifty"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print("NIFTY download failed:", meta["nifty"]["error"])
    (DATA / "snapshot.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
