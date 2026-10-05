# dpe — API Specification (cross-language contract)

This document is the normative contract for the `dpe` derivatives pricing
engine. The Python package under `python/src/dpe/` is the reference
implementation; the C++ (`namespace dpe`), Rust (crate `dpe`) and Java
(`com.quant.dpe`) ports MUST expose the public surface below with the same
semantics, units, edge-case behaviour and error behaviour, and MUST pass the
golden suite in `data/golden/golden.json` (schema and case list in §9).

Signatures are given in language-neutral pseudo-code. `float` means IEEE-754
double precision. Naming may follow each language's convention
(`bs_price` / `bsPrice` / `bs_price`), but parameter order and meaning are fixed.

---

## 1. Global conventions (apply to every function)

| Symbol  | Meaning | Unit / range |
|---------|---------|--------------|
| `s`     | spot price (equity) or FX rate quoted domestic-per-foreign | must be > 0 |
| `k`     | strike, same unit as `s` | must be >= 0 (`k = 0` allowed, see §3.1) |
| `t`     | time to expiry | **years**, must be >= 0 (`t = 0` = at expiry) |
| `sigma` | volatility | **decimal per sqrt-year** (0.20 = 20%), must be >= 0 |
| `r`     | risk-free rate (equity) / domestic rate `rd` (FX) | continuously compounded decimal; may be negative, must be finite |
| `q`     | dividend yield (equity) / foreign rate `rf` (FX) | same convention as `r` |
| `option_type` | `"call"` or `"put"` | case-insensitive on input; anything else is invalid |
| `style` | `"european"` or `"american"` | case-insensitive; anything else invalid |

Greek units (fixed; ports must NOT rescale):

* `delta` = dV/dS (dimensionless).
* `gamma` = d²V/dS².
* `vega`  = dV/dsigma **per unit of vol** (per 1.00, i.e. per 100 vol points).
* `theta` = dV/dt in **calendar time, per year** (= −dV/dT). Divide by 365
  for per-day theta; the engine never does that itself.
* `rho`   = dV/dr per unit of rate (`r` = domestic rate for FX). There is no
  reported dividend-rho / foreign-rho.
* `vanna` = d²V/(dS dsigma).
* `volga` = d²V/dsigma² (vomma).

Validation rules (identical everywhere): `s > 0`; `k >= 0`; `t >= 0`;
`sigma >= 0`; `r`, `q` finite; every input finite (NaN/inf always invalid).
Violations produce the language's invalid-input error (§8) with a message
naming the parameter.

---

## 2. Public types

```
Greeks {                     // analytic BSM/GK Greeks
  price, delta, gamma, vega, theta, rho, vanna, volga : float
}

BinomialGreeks {             // tree-based Greeks
  price, delta, gamma, theta : float
}

MCResult {                   // any Monte Carlo estimate
  value     : float          // point estimate (price or Greek)
  std_error : float          // standard error of the estimate
  ci_low    : float          // value - 1.959963984540054 * std_error
  ci_high   : float          // value + 1.959963984540054 * std_error
  n_paths   : int            // raw simulated paths (both antithetic halves)
}
```

Enumerations (may be enums or validated strings per language, but the
canonical string spellings are `"call"`, `"put"`, `"european"`,
`"american"`, `"crr"`, `"jr"`).

---

## 3. Analytic module (Black-Scholes-Merton & Garman-Kohlhagen)

### 3.1 `bs_price(s, k, t, sigma, r, q, option_type) -> float`

European BSM price with continuous dividend yield. With
`d1 = [ln(s/k) + (r − q + sigma²/2) t] / (sigma √t)`, `d2 = d1 − sigma √t`:

```
call = s e^{−qt} N(d1) − k e^{−rt} N(d2)
put  = k e^{−rt} N(−d2) − s e^{−qt} N(−d1)
```

Mandatory exact degenerate limits (checked by golden cases):

* `t = 0` → intrinsic: `max(±(s − k), 0)` (no discounting).
* `sigma = 0`, `t > 0` → discounted forward intrinsic:
  `call = max(s e^{−qt} − k e^{−rt}, 0)`, `put = max(k e^{−rt} − s e^{−qt}, 0)`.
* `k = 0` → `call = s e^{−qt}`, `put = 0`.

Implementation note: compute `N(x)` via `0.5 * erfc(−x/√2)` (not
`0.5*(1+erf)`) to keep deep-OTM tail precision; golden case
`bs_call_deep_otm` has tolerance 1e-10.

### 3.2 `bs_greeks(s, k, t, sigma, r, q, option_type) -> Greeks`

Full analytic Greeks (φ = standard normal pdf):

```
delta_call = e^{−qt} N(d1)             delta_put = e^{−qt}(N(d1) − 1)
gamma      = e^{−qt} φ(d1) / (s sigma √t)
vega       = s e^{−qt} φ(d1) √t
theta_call = −s e^{−qt} φ(d1) sigma/(2√t) − r k e^{−rt} N(d2) + q s e^{−qt} N(d1)
theta_put  = −s e^{−qt} φ(d1) sigma/(2√t) + r k e^{−rt} N(−d2) − q s e^{−qt} N(−d1)
rho_call   =  k t e^{−rt} N(d2)        rho_put  = −k t e^{−rt} N(−d2)
vanna      = −e^{−qt} φ(d1) d2 / sigma
volga      = vega · d1 · d2 / sigma
```

Degenerate region (`t = 0` or `sigma = 0` or `k = 0`) — contract values:
`gamma = vega = vanna = volga = 0`; with forward moneyness
`f = s e^{−qt} − k e^{−rt}`:

* call in the money (`f >= 0` or `k = 0`): `delta = e^{−qt}`,
  `theta = q s e^{−qt} − r k e^{−rt}`, `rho = k t e^{−rt}`; else all three 0.
* put in the money (`f < 0` and `k > 0`): `delta = −e^{−qt}`,
  `theta = r k e^{−rt} − q s e^{−qt}`, `rho = −k t e^{−rt}`; else all three 0.

**Warning (expiry-day convention).** Exactly at the money in this region
(`f = 0`) the call therefore reports `delta = e^{−qt}` and the put
`delta = 0` — the one-sided limit, not the 0.5 that many risk systems
assign to an expiring ATM option. This is a deliberate, tested tie-break
so that all four ports agree bit-for-bit; a desk that routes expiring
positions to a settlement workflow should do so before calling
`bs_greeks` with `t = 0`.

### 3.3 `gk_price(s, k, t, sigma, rd, rf, option_type) -> float`
### 3.4 `gk_greeks(s, k, t, sigma, rd, rf, option_type) -> Greeks`

Garman-Kohlhagen FX wrappers: **exactly** `bs_price`/`bs_greeks` with
`r = rd`, `q = rf`. Price is in domestic currency per unit of foreign
notional. The reported delta is the premium-excluded **spot delta**
`e^{−rf t} N(d1)`; `rho` is w.r.t. `rd`.

### 3.5 `implied_vol(price, s, k, t, r, q, option_type, tol = 1e-10, max_iter = 100) -> float`

Robust implied-vol inversion. Required behaviour:

1. Validate inputs; additionally require `t > 0` and `k > 0`.
2. No-arbitrage bounds with `lower = max(±(s e^{−qt} − k e^{−rt}), 0)`
   (sign per type) and `upper = s e^{−qt}` (call) / `k e^{−rt}` (put):
   if `price <= lower` raise/return error mentioning the **lower bound**
   ("below intrinsic"); if `price >= upper`, error mentioning the
   **upper bound**. No root-finding on inadmissible prices.
3. Otherwise find the unique root of `bs_price(sigma) = price` by
   safeguarded Newton (analytic vega) with bisection fallback inside a
   maintained bracket; initial bracket `[0, hi]` where `hi` doubles from
   1.0 (`1, 2, 4, 8, 16`, then clipped to the hard cap `20.0`) until the
   model price exceeds the target; if the model price at the cap is still
   below the target → error (the effective cap is exactly 20.0). Converged
   when `|model − price| < tol` **or** the bracket width is `< 1e-12`.
4. `tol` must be `> 0` and `max_iter >= 1` (else invalid-input error naming
   the parameter). `max_iter` counts Newton/bisection steps from the
   bracket midpoint; the residual is checked before every step and once
   after the last. If neither stopping rule is met, the function **fails**
   with a non-convergence error whose message contains `did not converge`,
   the residual `|model − price|` and the final bracket (see §8). A
   half-converged root is never returned as a number.
5. Accuracy contract: round-trips `sigma → price → sigma` to 1e-6 absolute
   for sigma in [0.05, 0.8] (golden cases `iv_roundtrip_*`), including deep
   ITM/OTM where Newton must not diverge (tiny vega ⇒ bisection). Vols up
   to the cap (e.g. 1.7, 17.0) round-trip through the bracket expansion.

---

## 4. Binomial module

### 4.1 `binomial_price(s, k, t, sigma, r, q, option_type, style, steps, method, richardson = false) -> float`

* `steps`: integer >= 1 (invalid otherwise). In Python any
  `numbers.Integral` (built-in `int`, `numpy.int32/int64`, ...) is
  accepted; `bool` and integral-valued floats (`500.0`) are invalid. The
  same holds for every integer parameter in §5 and §7.
* `method`: `"crr"` — `u = e^{sigma √dt}`, `d = 1/u`,
  `p = (e^{(r−q)dt} − d)/(u − d)`; `"jr"` — with
  `nu = (r − q − sigma²/2) dt`: `u = e^{nu + sigma √dt}`,
  `d = e^{nu − sigma √dt}`, `p = 1/2`. `dt = t / steps`.
* If the computed `p` falls outside (0, 1) (drift dominates vol at this
  step size), that is an invalid-input error, not a silent clamp.
* Terminal payoffs, discounted backward induction at `e^{−r dt}`; American
  style applies `value = max(continuation, intrinsic)` at **every** node.
* `richardson = true`: return the average of the `steps` and `steps + 1`
  prices (two-point odd/even Richardson averaging).
* Degenerate limits: `t = 0` → intrinsic. `sigma = 0` → European:
  discounted forward intrinsic (same as §3.1); American: maximum over the
  grid dates `t_i = i·t/steps`, `i = 0..steps`, of
  `e^{−r t_i} · payoff(s e^{(r−q) t_i})` (deterministic-path dynamic program).
* Overflow guard: the lattice is invalid (invalid-input error mentioning
  `overflow`, never `inf`/NaN) when
  `|ln s| + |(r − q − sigma²/2) t| + sigma √(t · n) > 700`, with
  `n = steps` (`steps + 1` when `richardson = true`). This bounds the
  extreme terminal log-spot of either lattice (the JR drift term is
  included for both methods so the admissible domain is method-independent).

### 4.2 `binomial_greeks(s, k, t, sigma, r, q, option_type, style, steps, method) -> BinomialGreeks`

Requires `steps >= 2`, `t > 0`, `sigma > 0` (else invalid-input error) and
the §4.1 overflow guard. `steps = 2` is legal and must return finite
values: level 2 is then the terminal level, which the rollback must retain
explicitly. With `V(i,j)`/`S(i,j)` the option value / spot at level `i`,
`j` up-moves, retained during a single backward induction:

```
delta  = (V(1,1) − V(1,0)) / (S(1,1) − S(1,0))
gamma  = [ (V(2,2)−V(2,1))/(S(2,2)−S(2,1)) − (V(2,1)−V(2,0))/(S(2,1)−S(2,0)) ]
         / ( (S(2,2) − S(2,0)) / 2 )
ds     = S(2,1) − s                       // 0 for CRR, O(dt) drift for JR
theta  = ( V(2,1) − V(0,0) − delta·ds − gamma·ds²/2 ) / (2 dt)
```

The `ds` correction strips the spot-displacement contribution out of the JR
calendar difference; it is exactly zero for CRR. `price` = `V(0,0)`.

---

## 5. Monte Carlo module

All MC functions: integer `n_paths >= 2` (>= 4 and even when
`antithetic = true`; odd + antithetic is invalid), integer `n_steps >= 1`
where present, integer `seed` in **`[0, 2^63 − 1]`** (a negative seed — or,
in Rust's `u64`, one above `2^63 − 1` — is an invalid-input error naming
`seed`; it is never silently wrapped or forwarded to the RNG library), and
require `t > 0`. Simulation uses the **exact** GBM step
`S_{t+h} = S_t · exp((r − q − sigma²/2) h + sigma √h Z)`.
Determinism: identical inputs + seed ⇒ identical output **within one
language** for a fixed toolchain (Python: numpy PCG64; C++: `mt19937_64`
bits through a fixed Box–Muller recipe, so libstdc++/libc++ agree up to
libm rounding; Rust: `StdRng` of the `rand` version in `Cargo.lock`; Java:
`SplittableRandom.nextGaussian`, specified since JDK 17). RNG streams are
NOT required to match across languages (§9.3).

Statistics contract: with antithetic pairs, the i.i.d. samples are the
**pair averages** `(f(Z_i) + f(−Z_i))/2` (n_paths/2 samples); `std_error` =
sample standard deviation (ddof = 1) of the i.i.d. samples divided by
√(sample count); CI multiplier is 1.959963984540054. `n_paths` in
`MCResult` is the raw path count.

Control-variate contract: adjusted samples `y − β (x − E[x])` with β =
sample-cov(y, x) / sample-var(x) estimated from the same run (β = 0 if
var(x) = 0); statistics computed on the adjusted samples.

### 5.1 `simulate_gbm_paths(s, t, sigma, r, q, n_paths, n_steps, seed, antithetic = false) -> matrix[n_paths][n_steps + 1]`

Column 0 = `s`; column i = `S(i · t / n_steps)`. With antithetic, rows
`[n_paths/2 ..)` use the negated normals of rows `[0 .. n_paths/2)`.

### 5.2 `mc_european(s, k, t, sigma, r, q, option_type, n_paths = 100000, seed = 42, antithetic = true, control_variate = false) -> MCResult`

Single exact step to `t`. Control variate = discounted terminal spot
`e^{−rt} S_T`, known mean `s e^{−qt}`.

### 5.3 `geometric_asian_price(s, k, t, sigma, r, q, option_type, n_fixings) -> float`

Closed form for the discretely monitored geometric-average Asian with
fixings at `t_i = i t / n`, `i = 1..n` (spot at 0 is NOT a fixing):

```
m = ln s + (r − q − sigma²/2) · t (n+1) / (2n)
v = sigma² t (n+1)(2n+1) / (6 n²)
EG = e^{m + v/2};  d1 = (ln(EG/k) + v/2)/√v;  d2 = d1 − √v
call = e^{−rt} (EG N(d1) − k N(d2));  put = e^{−rt} (k N(−d2) − EG N(−d1))
```

Requires `t > 0`, `k > 0`, integer `n_fixings >= 1`. If `v = 0` return
`e^{−rt} · max(±(EG − k), 0)`.

### 5.4 `mc_asian_arithmetic(s, k, t, sigma, r, q, option_type, n_paths = 100000, n_steps = 50, seed = 42, antithetic = true, control_variate = true) -> MCResult`

Arithmetic average over the same fixing grid as §5.3 (`i = 1..n_steps`).
Control variate = geometric-Asian payoff on the same paths, exact mean from
§5.3. Requires `k > 0`.

### 5.5 `mc_barrier_up_out(s, k, t, sigma, r, q, option_type, barrier, n_paths = 100000, n_steps = 100, seed = 42, antithetic = true) -> MCResult`

Discretely monitored up-and-out: pays vanilla payoff at `t` iff
`max_i S(t_i) < barrier` over ALL grid dates including `t_0 = 0`. Requires
`barrier > 0`. If `s >= barrier` return
`MCResult{0, 0, 0, 0, n_paths}` without simulating. This prices the
**discrete** contract; no Broadie-Glasserman-Kou continuity correction is
applied (document the O(1/√n_steps) high bias vs a continuous barrier).

### 5.6 `mc_lookback_floating(s, t, sigma, r, q, option_type, n_paths = 100000, n_steps = 100, seed = 42, antithetic = true) -> MCResult`

Floating-strike lookback over the grid (t = 0 included):
`call = e^{−rt}(S_T − min_i S(t_i))`, `put = e^{−rt}(max_i S(t_i) − S_T)`.
No strike parameter.

### 5.7 `mc_delta_pathwise(s, k, t, sigma, r, q, option_type, n_paths = 100000, seed = 42, antithetic = true) -> MCResult`

Pathwise delta of the European vanilla:
`call: e^{−rt} · mean(1{S_T > k} · S_T / s)`;
`put: −e^{−rt} · mean(1{S_T < k} · S_T / s)`.

### 5.8 `mc_delta_fd_crn(s, k, t, sigma, r, q, option_type, n_paths = 100000, seed = 42, rel_bump = 1e-4, antithetic = true) -> MCResult`

Central finite difference with common random numbers: reuse the SAME normal
draws for spots `s(1 ± rel_bump)`; per-path samples
`(payoff_up − payoff_down) / (2 · s · rel_bump)`, discounted. Requires
`0 < rel_bump < 1` (finite; both bumped spots must stay positive —
`rel_bump >= 1` is an invalid-input error naming `rel_bump`).

---

## 6. FX helpers

```
fx_forward(s, t, rd, rf) -> float            // s · e^{(rd − rf) t};  s > 0, t >= 0
spot_delta_to_forward(delta_spot, t, rf) -> float   // e^{+rf t} · delta_spot
forward_delta_to_spot(delta_forward, t, rf) -> float // e^{−rf t} · delta_forward
```

Premium-excluded conventions; sign-preserving (works for put deltas);
exact mutual inverses. `gk_greeks(...).delta` is a spot delta; forward
delta of a call is exactly `N(d1)`.

---

## 7. Utilities

```
historical_vol(prices: array of float, periods_per_year: int = 252) -> float
```

Annualised close-to-close realised vol: sample std (ddof = 1) of
`ln(P_i / P_{i−1})` times `√periods_per_year`. Errors: fewer than 3 prices;
any price <= 0 or non-finite; `periods_per_year` not an integer >= 1.

---

## 8. Error behaviour

Every violation described above is an *invalid-input error* whose message
names the offending parameter or bound. The implied-vol solver has two
further failure classes — a no-arbitrage violation (§3.5 step 2, or the
vol cap) and non-convergence (§3.5 step 4) — which use the same mechanism
in Python/C++/Java and dedicated variants in Rust:

| Language | Invalid input | No-arbitrage / cap | Non-convergence |
|----------|---------------|--------------------|-----------------|
| Python   | `ValueError` | `ValueError` (message: `lower bound` / `upper bound` / `exceeds cap`) | `ValueError` (message: `did not converge`) |
| C++      | `std::invalid_argument` | `std::invalid_argument` | `std::invalid_argument` |
| Rust     | `Err(DpeError::InvalidInput(msg))` | `Err(DpeError::NoArbitrage(msg))` | `Err(DpeError::NoConvergence(msg))` |
| Java     | `IllegalArgumentException` | `IllegalArgumentException` | `IllegalArgumentException` |

Rust never panics on bad input: every public function returns
`Result<_, DpeError>` (a manual thiserror-style enum with exactly the three
variants above). A `matches!(e, DpeError::InvalidInput(_))` guard therefore
does **not** catch bound violations or non-convergence; match all three or
use the `Display` message.

No pricing function ever returns NaN/inf for valid inputs; invalid inputs
never return a number; an iterative solver never returns an unconverged
result as if it had converged.

---

## 9. Golden suite (`data/golden/golden.json`)

### 9.1 Schema

```json
{ "cases": [ { "name": "...", "inputs": { flat scalars }, "expect": { flat scalars }, "tol": 1e-8 } ] }
```

`tol` is an **absolute** tolerance applied to every key in `expect`
independently. Expected values are stored at full double precision (the
generator does not round them, so the whole of `tol` is available to the
port). Option type is in `inputs.type`; FX cases use `rd`/`rf` instead of
`r`/`q`. Every port's test suite must load this file, assert each case,
and fail on a case name it does not know how to dispatch; suites must not
pin the exact case count (they assert `>= 20`).

### 9.2 Case list and the function each exercises

| Case name | Function under test | Expect keys |
|-----------|--------------------|-------------|
| `bs_call_atm` | `bs_price` | `price` |
| `bs_put_atm` | `bs_price` | `price` |
| `bs_call_dividend` | `bs_price` (q = 2%) | `price` |
| `bs_put_deep_itm` | `bs_price` | `price` |
| `bs_call_deep_otm` | `bs_price` (tail precision, tol 1e-10) | `price` |
| `bs_call_high_vol` | `bs_price` (sigma = 80%) | `price` |
| `bs_put_weekly` | `bs_price` (t = 1/52) | `price` |
| `bs_call_leaps` | `bs_price` (t = 3) | `price` |
| `bs_call_sigma_zero` | `bs_price` (sigma = 0 limit) | `price` |
| `bs_put_expiry` | `bs_price` (t = 0 limit) | `price` |
| `bs_call_zero_strike` | `bs_price` (k = 0 limit) | `price` |
| `bs_greeks_call_atm` | `bs_greeks` | `delta gamma vega theta rho vanna volga` |
| `bs_greeks_put_itm` | `bs_greeks` | `delta gamma vega theta rho vanna volga` |
| `gk_call_eurusd` | `gk_price`, `gk_greeks`, `spot_delta_to_forward` | `price delta_spot delta_forward` |
| `gk_put_negative_rate` | `gk_price` (rd = −0.5%) | `price` |
| `iv_roundtrip_call_atm` | `implied_vol` | `sigma` |
| `iv_roundtrip_put_otm` | `implied_vol` | `sigma` |
| `crr_euro_call_2000` | `binomial_price` (crr, european, 2000) | `price` |
| `crr_amer_put_500` | `binomial_price` (crr, american, 500) | `price` |
| `crr_amer_put_div_500` | `binomial_price` (crr, american, q = 3%) | `price` |
| `jr_euro_put_500` | `binomial_price` (jr, european, 500) | `price` |
| `crr_greeks_call_500` | `binomial_greeks` (crr, 500) | `delta gamma theta` |
| `geo_asian_call_analytic` | `geometric_asian_price` (12 fixings) | `price` |
| `mc_euro_call_100k` | `mc_european` (antithetic) | `price` |
| `mc_asian_call_cv_12fix` | `mc_asian_arithmetic` (antithetic + CV) | `price` |
| `mc_barrier_single_step_call` | `mc_barrier_up_out` (antithetic, `n_steps = 1`, `barrier` in inputs) | `price` |
| `mc_lookback_single_step_call` | `mc_lookback_floating` (antithetic, `n_steps = 1`) | `price` |
| `mc_lookback_single_step_put` | `mc_lookback_floating` (antithetic, `n_steps = 1`) | `price` |

28 cases in total.

### 9.3 Monte Carlo cases — tolerance semantics

RNG streams are language-specific, so for the five `mc_*` cases the
`inputs.seed` is **advisory** (use any fixed seed of your RNG; keep the run
deterministic). The expected values are analytic (`mc_euro_call_100k` = the
Black-Scholes price; `mc_barrier_single_step_call` =
`C(k) − C(B) − (B − k) e^{−rt} N(d2(B))`, the one-monitoring-date
up-and-out; `mc_lookback_single_step_{call,put}` = the ATM vanilla, since
with one step the floating-strike payoff is `(S_T − s)^+` / `(s − S_T)^+`)
or a high-precision reference (`mc_asian_call_cv_12fix` = a 200k-path CV
run; its own SE < 0.005). The tolerances (0.15, 0.05, 0.06, 0.10, 0.07) are
several standard errors wide (≈ 4–5 SE) at the stated `n_paths` with the
stated variance reduction; a correct implementation at the stated
`n_paths` passes with overwhelming probability. Deterministic cases
(everything else) carry tight tolerances (1e-8 / 1e-10 / 1e-6) and MUST
match bit-for-bit up to floating-point noise.

### 9.4 Regeneration

`python3 data/generate_data.py` regenerates `portfolio.csv`,
`spots_timeseries.csv` (fixed seed 20260827) and `golden.json`, after
re-validating the Python implementation against analytic anchors,
put-call parity, implied-vol round-trip, tree convergence and an MC
3-standard-error check. Ports never regenerate the file; they only consume it.

## 9.5 Conformance verification

The numerical contract in this document is language-neutral; build systems and
test frameworks are deliberately **non-normative**. The C++ port verifies this
contract with CMake, GoogleTest/CTest and the shared golden file. Optional
ASan+UBSan builds exercise the same tests under runtime memory/undefined-behavior
checks. Benchmark and profiling targets measure performance only; they do not
change pricing semantics or tolerances. See [`cpp/README.md`](cpp/README.md).

Repository documentation and build commands are always relative to the repository
root. Generated CMake directories such as `cpp/build`, `cpp/build-sanitize`,
`cpp/build-bench` and `cpp/build-profile` are local artifacts and must not be
committed.

---

## 10. Data files

`data/portfolio.csv` — columns
`id, underlying_type, s, k, t, sigma, r, q, type, style, quantity`;
`underlying_type` in {`equity`, `fx`}; for `fx` rows `r` = rd and `q` = rf.
Quantities may be negative (short positions). One row (`EQ_AMER_P1`) is
American style and must be priced on the tree (500 steps CRR) in demos.

`data/spots_timeseries.csv` — columns `date, ACME, GLOBEX, PIPCO`;
505 rows of daily closes for the historical-vol demo
(`historical_vol` with 252 periods/year).
