# Learning log

Notes written while building the engine, in the order things were done. Each step has
**what**, **why**, and **what to say in an interview**. Read this before talking about the
project; you should be able to explain every line of it.

---

## Step 0. The big picture

**The question.** Black-Scholes says one number, the volatility, describes an option
market. Real markets disagree: every strike and expiry trades at a different implied vol
(the *smile*, or for equity indices the *skew*: low strikes are expensive). The Heston
model explains the smile by making variance itself random. This project builds a Heston
engine the way a bank's quant library is built: several independent pricing methods,
each tested against the others, then calibrated to real option prices.

**The pipeline.**

```
Heston model --> characteristic function --> Fourier pricers (exact, fast)
            \--> SDE simulation ----------> Monte Carlo (any payoff)
             \-> PDE -----------------------> finite differences (American options, Greeks)
all three must agree --> calibrate to market smile --> Greeks / hedge ratios
```

**Why three method families?** Each one covers what the others can't:

| Method | Strength | Weakness | Used for |
|---|---|---|---|
| Fourier (COS, FFT) | microseconds per price, machine accuracy | European payoffs only | calibration |
| Monte Carlo | any payoff, path dependence, high dimensions | slow, noisy, discretisation bias | exotics, xVA, risk |
| Finite differences (PDE) | early exercise, all Greeks from one solve | curse of dimensionality (2-3 factors max) | American/Bermudan, barriers |

In practice a desk calibrates with Fourier, then prices the exotic with MC or PDE on the
calibrated parameters. Cross-checking them is what model validation teams do.

**Link to the next project.** The Monte Carlo simulator here, with calibrated parameters,
is the market simulator for deep hedging.

---

## Step 1. Black-Scholes baseline (`src/heston/black_scholes.py`)

**What.** Price, analytic Greeks (delta, gamma, vega, theta, rho) and implied volatility.

**Why.** Everything in options is quoted in Black-Scholes implied vol, even when the model
behind it is Heston. And Heston must reduce to Black-Scholes when vol-of-vol is zero, which
is the first test of the Heston code.

**Implied vol solver.** Newton-Raphson on sigma, with vega as the derivative:
`sigma <- sigma - (BS(sigma) - price) / vega`. Quadratic convergence (digits double each
step) from a good start: the Brenner-Subrahmanyam approximation `sigma ~ sqrt(2 pi / T) C / S`
for near-the-money options. Newton fails when vega is tiny (deep in or out of the money),
so the code falls back to Brent's method, a bracketing root finder that always converges
because the BS price is strictly increasing in sigma. Prices outside the no-arbitrage
bounds (a call above spot, or below intrinsic) return `nan`: there is no vol for them.

**Tests.** Hull's textbook example, put-call parity, every Greek against a numerical
derivative, and implied vol round trips across strikes and vols.

**Interview lines.**
- "How do you compute implied vol?" Newton with vega, good initial guess, Brent fallback,
  and say *why* the fallback is needed.
- "Why can't you get an implied vol for a deep ITM option with 1e-10 of time value?" The
  price carries no information about vol: vega is ~0.

---

## Step 2. The Heston model (`src/heston/model.py`)

**What.**

```
dS = (r - q) S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + sigma sqrt(v) dW2,     corr(dW1, dW2) = rho
```

| Parameter | Meaning | What it does to the smile |
|---|---|---|
| v0 | today's variance | level of short-dated vols |
| theta | long-run variance | level of long-dated vols |
| kappa | mean-reversion speed | how fast short-dated vols move to long-dated |
| sigma | vol of variance | curvature (smile convexity, fat tails) |
| rho | spot-vol correlation | skew; equities have rho ~ -0.7 (crashes come with vol spikes) |

See `results/smile_parameters.png` for rho tilting the smile and sigma bending it.

**Feller condition.** If `2 kappa theta >= sigma^2`, the variance never reaches 0.
Calibrated equity parameters almost always *violate* it, so every numerical method has to
cope with v hitting 0. The Fang-Oosterlee test case used throughout has Feller ratio 0.38
on purpose.

**The characteristic function.** `phi(u) = E[exp(i u log(S_T/S_0))]`. For Heston it is
known in closed form (it is "affine": the exponent is linear in v0). Every Fourier method
needs only this one function.

**The "little Heston trap".** Heston's 1993 formula contains `log((1 - g e^{-dT}) / (1 - g))`.
The complex logarithm has branch cuts, and in the original parameterisation the argument
crosses the cut for long maturities, so the integrand jumps and prices are silently wrong.
Albrecher et al. (2007) showed that writing it with `g -> 1/g` (which we use) stays on
the principal branch. A test checks phi is continuous in u for T = 30 years.

**Sanity checks.** `phi(0) = 1` and `phi(-i) = e^{(r-q)T}`: the second is the martingale
condition, `E[S_T] = S_0 e^{(r-q)T}` under the risk-neutral measure.

**Interview lines.**
- "What does rho do in Heston?" Controls the skew; explain why it's negative for equities
  (leverage effect: falling prices raise volatility).
- "What's the Feller condition and does it matter?" Yes: it decides whether the variance
  can hit zero, which breaks naive simulation schemes (see step 4).

---

## Step 3. Fourier pricing (`src/heston/fourier.py`)

Three ways to turn the characteristic function into an option price.

**3a. Gil-Pelaez / Heston's original form.** A call is `S e^{-qT} P1 - K e^{-rT} P2`, where
P2 is the risk-neutral probability of finishing in the money and P1 the same probability
under the "stock measure". The Gil-Pelaez inversion theorem gives each as an integral of
the characteristic function. Adaptive quadrature (`scipy.integrate.quad`) makes it very
accurate: this is our reference value. Result: 5.785155434 vs the published 5.785155450.

**3b. Carr-Madan FFT.** The call price as a function of log strike isn't integrable (it
tends to S as K -> 0), so multiply by `e^{alpha k}` with alpha ~ 1.5 (damping). The damped
price's Fourier transform is a closed-form expression in phi. Sampling the inverse
transform on a grid turns it into one FFT that prices 4096 strikes at once; Simpson
weights improve the quadrature. Bug I hit: the strike grid has to be centred at log S *in
the FFT phase factor too*, not just in the strike labels; first version gave 0.099
instead of 5.785.

**3c. COS method (Fang & Oosterlee 2008).** Truncate the density of `y = log(S_T/K)` to an
interval [a, b], expand it in a cosine series. The cosine coefficients are just the
characteristic function at `u_k = k pi / (b - a)`, and the payoff's cosine coefficients are
closed form, so the price is a sum of N terms. Error falls *exponentially* in N
(`results/fourier_convergence.png`): N = 128 gives 1e-5, N = 256 gives 1e-7, in 0.15 ms.

Choosing [a, b] is the whole art:
- Centre at the mean of y, `log(S/K) + c1`, separately for each strike.
- Half-width `L sqrt(c2 + sqrt(c4))` with L = 10. My first version used the variance c2
  alone and had a 4e-5 error that no amount of N fixed: with sigma = 0.58 the tails are
  fat, and the fourth cumulant c4 is what measures that. The cumulants are computed by
  finite differences of `log phi` at 0, so no messy cumulant formulas are needed.
- Price puts, not calls, then use put-call parity. The put payoff is bounded, so
  truncating the interval costs little; a call's payoff grows like e^y, and truncation
  errors blow up.
- Edge cases found by tests: for a deep ITM put at very short maturity the whole interval
  sits below 0, and for a deep OTM one above 0. The payoff integration bound must be
  `min(b, 0)` and an empty interval contributes nothing.

**Interview lines.**
- "How would you price 10,000 options fast for calibration?" COS or FFT: explain why
  (one characteristic function evaluation per term, vectorised over strikes).
- "Why price puts and convert?" Bounded payoff, robust to truncation.
- "Why the damping factor in Carr-Madan?" Integrability.

---

## Step 4. Monte Carlo (`src/heston/monte_carlo.py`)

**Euler with full truncation.** Discretise both SDEs with step dt. The problem: Euler can
make v negative, and then `sqrt(v)` fails. Full truncation uses `max(v, 0)` inside drift and
diffusion (Lord et al. 2010), the least biased of the simple fixes. Correlated normals:
`Z2 = rho Z1 + sqrt(1 - rho^2) Z`.

**Andersen QE (2008).** Over one step, v_{t+dt} given v_t has a known mean and variance
(it's a CIR process). QE samples from a distribution with exactly those moments: a squared
normal when variance is high relative to the mean (psi <= 1.5), or a point mass at zero
plus an exponential tail when v is close to zero. The log-spot step integrates v with the
trapezoid rule and recovers `int sqrt(v) dW2` from the variance equation, so the spot-vol
correlation is handled exactly.

**Result** (`results/mc_bias.png`, 1M paths): at 4 steps per year Euler is off by 0.54 (a
9% price error), QE by 0.004. Euler needs ~64 steps to get to QE-at-2-steps accuracy. Bias
doesn't shrink with more paths; only with smaller dt. Standard error does shrink with paths
(like 1/sqrt(n)). Two different errors; know which one you're looking at.

**Variance reduction.**
- *Antithetic variates:* use Z and -Z. Errors computed on pair averages, since the two
  halves of a pair are not independent.
- *Control variate:* the discounted S_T has a known mean (S0 e^{-qT}), and payoffs are
  correlated with it. Subtract `beta (control - known mean)`: 63% less variance on a
  K = 90 call, for free.

**Interview lines.**
- "Your MC price is off by 3 standard errors. Why?" Discretisation bias, not noise: halve
  dt and see if it moves.
- "How do you simulate a CIR/Heston variance?" Explain why Euler fails when Feller is
  violated, then QE.

---

## Step 5. Finite differences in 1D (`src/heston/fd1d.py`)

**The PDE.** In log spot x and time to expiry tau, Black-Scholes is
`V_tau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V`. Central differences turn the
right-hand side into a tridiagonal matrix A.

**theta-scheme.** `(I - theta dt A) V^{n+1} = (I + (1 - theta) dt A) V^n`.
- theta = 0, explicit: no linear solve, but unstable unless `dt <= dx^2 / sigma^2`.
- theta = 1, implicit Euler: always stable, first order in time.
- theta = 1/2, Crank-Nicolson: second order. Confirmed in the results table: halving the
  grid divides the CN error by 4.00 and the implicit error by ~2.

**Rannacher smoothing.** CN is stable but doesn't damp high-frequency errors, and the
payoff kink at K is all high frequencies. With big time steps the error rings around the
strike, and Gamma (the second derivative) shows it badly: `results/rannacher_gamma.png`
shows Gamma spiking to 5.5 when the true value is 0.06. Fix: start with a few implicit
Euler half-steps (which damp strongly), then switch to CN. Gamma becomes smooth and the
price is still second order. The strike is also placed exactly on a grid node.

**American put.** Early exercise means `V >= K - S` everywhere. Each time step becomes a
linear complementarity problem: either the PDE holds, or V equals the payoff. Projected
SOR solves it by Gauss-Seidel sweeps that clip V to the payoff after every update.
Result: 6.0900 vs the literature value 6.0904 (the European put is 5.57; the 0.52 difference
is the early-exercise premium).

**Interview lines.**
- "Why does Crank-Nicolson give bad Greeks sometimes?" Non-smooth payoff + no damping;
  Rannacher.
- "How do you price an American option on a PDE?" LCP, PSOR (or penalty methods,
  Brennan-Schwartz for puts).

---

## Step 6. The Heston PDE with ADI (`src/heston/fd_adi.py`)

**The PDE.** Two space dimensions now, S and v:

```
u_tau = 0.5 v S^2 u_SS + rho sigma v S u_Sv + 0.5 sigma^2 v u_vv
        + (r - q) S u_S + kappa (theta - v) u_v - r u
```

**Why not just implicit?** On a 200 x 100 grid, an implicit step is a 20,000-unknown sparse
system with a messy band structure (the mixed term u_Sv couples diagonal neighbours).
ADI (alternating direction implicit) splits the operator into A0 (mixed), A1 (S terms)
and A2 (v terms), treats A1 and A2 implicitly *one direction at a time* (each solve only
couples points along one grid line), and A0 explicitly. The three schemes differ in how
they correct the error from the explicit A0:

| Scheme | theta | Order in time |
|---|---|---|
| Douglas | 1/2 | 1 when rho != 0 (the explicit A0 term spoils it) |
| Craig-Sneyd | 1/2 | 2 |
| Hundsdorfer-Verwer | 1/2 + sqrt(3)/6 | 2, and most robust |

Measured (`results/validation.md`): doubling the time steps cuts Douglas's error ~2x and
CS/HV's ~4x. CS with only 10 steps has a 3e-2 error (it is less stable at large steps);
HV doesn't. Tests check these orders, not just that prices are close.

**Grid.** Non-uniform, from In 't Hout & Foulon (2010): `S = K + c sinh(xi)` packs points
around the strike; `v = d sinh(eta)` packs them near v = 0, where the solution bends most.
Derivatives use 3-point stencils for non-uniform spacing.

**Boundaries.** S = 0: call worth 0. S_max = 8K: deep in the money, `S e^{-q tau} - K e^{-r tau}`.
v_max = 5: huge variance, call worth S. At v = 0 there is *no* boundary condition: the PDE
degenerates to a first-order equation there and we just discretise it with a one-sided
derivative (this is the Feller story again, from the PDE side).

**Results.** 200 x 100 grid: 6e-4 error vs COS in 0.3 s, and spatial error falls at
second order. American puts (with projection after each step) match the Clarke-Parrott /
Ikonen-Toivanen benchmark values to ~4e-4.

**What I tried and turned off.** Damping (Rannacher-style implicit start) hurt here: it
adds a first-order error to the price and HV/CS were already smooth, so the default is off.

**Interview lines.**
- "How would you solve a 2D PDE efficiently?" ADI, explain the splitting and the mixed
  derivative problem.
- "What boundary condition do you put at v = 0?" None: the PDE itself, degenerate.

---

## Step 7. Validation (`scripts/02_validate_methods.py`, `results/validation.md`)

One option, seven methods:

| Method | Error vs reference | Time |
|---|---|---|
| Gil-Pelaez integral | 2e-8 | 1.7 ms |
| Carr-Madan FFT | 2e-7 | 0.8 ms |
| COS, N = 256 | 8e-8 | **0.15 ms** |
| MC Euler, 100 steps, 400k paths | 1.5e-2 | 1.4 s |
| MC QE, 100 steps, 400k paths | 2e-3 (within its std error 8e-3) | 3.2 s |
| ADI HV, 200 x 100 x 100 | 6e-4 | 0.26 s |
| ADI CS, 200 x 100 x 100 | 6e-4 | 0.30 s |

**What this shows.** COS is about 10,000 times faster than MC and five orders of magnitude
more accurate, which is why calibration uses it. The reason to have MC and PDE at all is
the payoffs COS can't do (path-dependent, early exercise).

**Interview line.** "How do you know your pricer is right?" Independent methods agree,
convergence orders match theory, limits (BS when sigma -> 0) hold, published benchmarks match.

---

## Step 8. Calibration (`src/heston/calibration.py`, `scripts/03_calibrate.py`)

**Data.** SPX option chains from Yahoo (and NIFTY from NSE when it doesn't block the
runner), downloaded by the `download-data` workflow and committed to `data/`. Both are
European-style index options, so they match the model.

**Cleaning.** Drop zero bids and wide spreads (> 25% of mid). Keep 2 weeks to 2 years.
Back out the forward F and discount factor D per expiry from put-call parity,
`C - P = D (F - K)`, by regressing C - P on K near the money: no need to guess dividends.
Keep out-of-the-money options only.

**Objective.** `sum ((model - market) / vega)^2`: a price error divided by vega is
(approximately) an implied-vol error, so cheap wing options and expensive ATM ones are
weighted fairly. Least squares with bounds (`scipy.optimize.least_squares`, trust region)
from four starting points; keep the best.

**Ill-posedness.** kappa and sigma trade off (both control how the smile flattens with
maturity), so different starts can find different kappa with almost the same error. Say
this in interviews. Regularisation (penalising changes from yesterday's parameters) is what
desks do for stability.

**Check on synthetic data.** Prices from known parameters plus bid-ask noise: calibration
recovers v0, kappa, theta, sigma, rho to 3 significant digits, 0.04 vol points RMSE.

**Real-data results:** see `results/calibration.md` (filled in after the first download).

---

## Step 9. Greeks and hedge ratios (`src/heston/greeks.py`, `results/greeks.md`)

Delta three ways, all agreeing: COS bump-and-reprice, the ADI grid (one solve gives delta
at every spot), and Monte Carlo pathwise (`dPayoff/dS0 = 1{S_T > K} S_T / S0`).

**The interesting part: which delta?** For an ATM call with rho = -0.7:

| Hedge ratio | Value | Assumes |
|---|---|---|
| Black-Scholes delta at implied vol | 0.572 | implied vol stays put when spot moves ("sticky strike") |
| Heston delta (v fixed) | 0.687 | variance doesn't move with spot |
| Heston minimum-variance delta | 0.484 | variance moves with spot as the correlation says |

Heston delta is *above* BS because Heston prices depend on K/S: when spot rises, the option
moves to a lower K/S on a downward-sloping skew, so its implied vol rises. The
minimum-variance delta, `delta + (rho sigma / S) dC/dv` (Bakshi, Cao & Chen 1997), is
*below* both: with rho < 0, spot up means variance probably down, which hurts a call.

Which one actually hedges best is an empirical question. That's exactly what the deep
hedging project tests, with a network that learns the hedge directly.

---

## How to talk about this project

**Resume line (fill in calibration numbers after the real-data run):**
"Built a Heston stochastic-volatility pricing engine in Python: Fourier (COS, Carr-Madan
FFT), Monte Carlo (Andersen QE) and 2D finite-difference (ADI) pricers cross-validated to
1e-4 against published benchmarks; calibrated to SPX options with x vol-point RMSE."

**Questions to expect.**
- Derive put-call parity. Why is the BS price increasing in vol?
- What is a characteristic function and why does Fourier pricing need it?
- Why does Euler fail for the CIR process? What is the Feller condition?
- What is the stability condition for the explicit scheme? Why is CN unconditionally stable?
- What does each Heston parameter do to the smile? Why is rho negative for equities?
- Why can't Heston fit very short-dated smiles well? (Diffusion can't produce enough
  short-term skew; jumps (Bates) or rough volatility can.)
- Is calibration unique? (No: kappa/sigma trade-off.)
