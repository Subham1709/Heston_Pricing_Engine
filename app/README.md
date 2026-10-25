# Web app

Live: https://subham-heston-engine.streamlit.app (Streamlit Community Cloud, main file
`app/streamlit_app.py`, requirements in `app/requirements.txt`).

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The sidebar holds the five Heston parameters (start: the SPX calibration). Every tab
recomputes live with the engine in `src/heston`.

| Tab | What happens |
|---|---|
| Price an option | Prices one European option by COS, Carr-Madan FFT and the Gil-Pelaez integral (and optionally Monte Carlo QE and the ADI PDE), with timings and differences. |
| Smile | Turns Heston prices into Black-Scholes implied vols across strikes for four maturities, so you can see what each parameter does to the smile. |
| SPX calibration | Market vs calibrated smile for any of the 13 expiries (from `results/calibration_spx.csv`); optionally your sidebar parameters on the same quotes. |
| Hedge ratios | BS delta at implied vol, Heston delta and minimum-variance delta across strikes (`cos_greeks`). |
| Validation | The seven-method benchmark table and convergence figures from `results/`. |

Only the calibration itself is not rerun (about 100 s); run `scripts/03_calibrate.py` for that.
