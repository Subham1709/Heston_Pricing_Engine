# Greeks under Heston

Parameters: v0 = 0.04, kappa = 1.5, theta = 0.04, sigma = 0.6, rho = -0.7; S = 100.0, T = 0.5, r = 0.03. Calls.

|        K |   implied vol |   delta COS bump |   delta ADI grid |   delta MC pathwise |   MC std err |   min-variance delta |   BS delta at implied vol |   gamma COS |   gamma ADI |
|---------:|--------------:|-----------------:|-----------------:|--------------------:|-------------:|---------------------:|--------------------------:|------------:|------------:|
|  80.0000 |        0.2614 |           0.9489 |           0.9489 |              0.9495 |       0.0004 |               0.8834 |                    0.9163 |      0.0039 |      0.0039 |
|  90.0000 |        0.2227 |           0.8717 |           0.8718 |              0.8725 |       0.0005 |               0.7422 |                    0.8004 |      0.0103 |      0.0103 |
| 100.0000 |        0.1815 |           0.6870 |           0.6872 |              0.6870 |       0.0006 |               0.4835 |                    0.5718 |      0.0282 |      0.0282 |
| 110.0000 |        0.1464 |           0.2889 |           0.2881 |              0.2891 |       0.0007 |               0.1294 |                    0.2345 |      0.0469 |      0.0467 |
| 120.0000 |        0.1384 |           0.0464 |           0.0468 |              0.0464 |       0.0004 |               0.0092 |                    0.0484 |      0.0108 |      0.0109 |

The three Heston deltas agree (COS bump and the ADI grid to ~1e-3, MC within its error bars).

![](delta_vs_bs.png)

Three different numbers for 'the' delta:

- **Black-Scholes delta at the option's implied vol** assumes implied vol stays put when spot moves.
- **Heston delta** holds the variance v fixed. Heston prices depend on K/S, so when spot rises the option moves to a lower K/S, where the (negatively skewed) smile is higher: implied vol rises with spot, and this delta is *above* Black-Scholes.
- **Minimum-variance delta** = delta + rho sigma / S * dC/dv. In Heston, v tends to fall when spot rises (rho < 0), which lowers the call's value, so the best hedge holds *less* stock than the v-fixed delta.

Which one hedges best on real paths is an empirical question, and it is the starting point of the deep hedging project, where a neural network learns the hedge ratio directly.
