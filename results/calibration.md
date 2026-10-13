# Calibration to market option prices

Snapshot downloaded 2026-10-13T11:19:38+00:00 (UTC).

## SPX

Spot 7807.24. 1880 out-of-the-money quotes over 13 expiries (16 to 465 days), K/F in (0.75, 1.2). Calibrated in 96.9 s.

| parameter | value |
|---|---|
| v0 (vol 11.6%) | 0.0136 |
| kappa (half-life 1.1 months) | 7.408 |
| theta (vol 20.5%) | 0.0421 |
| sigma | 1.895 |
| rho | -0.648 |
| Feller ratio 2 kappa theta / sigma^2 | 0.17 |

RMSE in implied vol: **0.90 vol points** overall.

![](calibration_spx.png)

Fit by expiry (funding rate r and dividend yield q implied from put-call parity):

|   days |   quotes |   rmse_vol_pts |     r |     q |
|-------:|---------:|---------------:|------:|------:|
|     16 |      130 |          1.781 | 0.048 | 0.036 |
|     21 |      110 |          1.372 | 0.048 | 0.030 |
|     24 |      193 |          1.503 | 0.048 | 0.025 |
|     29 |       84 |          1.116 | 0.048 | 0.026 |
|     34 |       49 |          1.096 | 0.048 | 0.025 |
|     45 |      102 |          0.810 | 0.048 | 0.022 |
|     66 |      355 |          0.591 | 0.048 | 0.020 |
|    108 |      176 |          0.427 | 0.048 | 0.013 |
|    157 |      123 |          0.338 | 0.048 | 0.013 |
|    220 |      121 |          0.263 | 0.048 | 0.011 |
|    276 |      122 |          0.319 | 0.048 | 0.010 |
|    352 |      201 |          0.333 | 0.048 | 0.009 |
|    465 |      114 |          0.490 | 0.048 | 0.007 |

## NIFTY

No data: JSONDecodeError: Expecting value: line 1 column 1 (char 0)
