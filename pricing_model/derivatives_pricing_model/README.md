# Derivatives Pricing Engine

A four-language (Python / C++ / Rust / Java) pricing library for vanilla and
first-generation exotic options, built around three classic engines:

1. **Analytic Black-Scholes-Merton** (equity with continuous dividend yield)
   and **Garman-Kohlhagen** (FX, domestic/foreign rates) — prices, the full
   Greek set (delta, gamma, vega, theta, rho, **vanna**, **volga**) and a
   robust bracketed-Newton **implied-volatility** solver.
2. **Binomial trees** — Cox-Ross-Rubinstein and Jarrow-Rudd lattices,
   European **and American** exercise, tree Greeks, optional two-point
   Richardson averaging.
3. **Monte Carlo** — exact-step GBM simulation; European vanillas,
   arithmetic Asians, up-and-out barriers, floating-strike lookbacks;
   antithetic and control variates; standard errors and 95% confidence
   intervals; pathwise and common-random-numbers finite-difference deltas.

The Python package under `python/src/dpe/` is the **reference
implementation**. The C++ (`namespace dpe`), Rust (crate `dpe`) and Java
(`com.quant.dpe`) ports implement the same contract, defined normatively in
[`API_SPEC.md`](API_SPEC.md), and all four are pinned together by a shared
golden-value suite (see below).

Documentation map:

| File | What it covers |
|------|----------------|
| [`API_SPEC.md`](API_SPEC.md) | Normative cross-language contract: signatures, units, edge cases, validation and errors |
| [`LEARN.md`](LEARN.md) | Theory: BSM/GK derivations, Greeks, implied vol, trees, Monte Carlo and FX conventions |
| [`COOKBOOK.md`](COOKBOOK.md) | Task-oriented recipes with copy-paste examples for Python, C++, Rust and Java |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component design, data flow, numerical decisions, testing strategy and engineering toolchain |
| [`docs/GITHUB_PAGES.md`](docs/GITHUB_PAGES.md) | How to publish the static documentation site with GitHub Pages |
| [`cpp/README.md`](cpp/README.md) | C++ build, GoogleTest/CTest, sanitizers, benchmarks, profiling and CI commands |

## Feature matrix

| Capability | Python | C++ | Rust | Java |
|---|---|---|---|---|
| BSM price (dividend yield `q`) | yes | yes | yes | yes |
| Garman-Kohlhagen FX (`rd`/`rf`) | yes | yes | yes | yes |
| Greeks incl. vanna/volga | yes | yes | yes | yes |
| Implied vol (bracketed Newton + bisection) | yes | yes | yes | yes |
| Binomial CRR + Jarrow-Rudd, European/American | yes | yes | yes | yes |
| Tree Greeks (delta, gamma, theta) | yes | yes | yes | yes |
| Richardson averaging | yes | yes | yes | yes |
| MC European (antithetic, control variate) | yes | yes | yes | yes |
| MC arithmetic Asian (geometric-Asian control) | yes | yes | yes | yes |
| MC up-and-out barrier (discrete monitoring) | yes | yes | yes | yes |
| MC floating-strike lookback | yes | yes | yes | yes |
| MC delta: pathwise + FD with common random numbers | yes | yes | yes | yes |
| FX helpers (forward, spot/forward delta conversion) | yes | yes | yes | yes |
| Historical (realised) volatility | yes | yes | yes | yes |
| Golden-value cross-language tests | yes | yes | yes | yes |
| Exact `t = 0` / `sigma = 0` / `k = 0` limits | yes | yes | yes | yes |
| Negative-rate support (JPY/CHF-style) | yes | yes | yes | yes |
| Implied vol: `tol`/`max_iter` validated, non-convergence is an error (never a half-converged root) | yes | yes | yes | yes |
| Uniform input contract: seed in `[0, 2^63−1]`, `rel_bump ∈ (0,1)`, lattice overflow guard, `steps = 2` tree Greeks | yes | yes | yes | yes |
| numpy/pandas integer types accepted for counts and seeds | yes | n/a | n/a | n/a |

## Directory layout

```
derivatives-pricing-engine/
  README.md                # this file
  LEARN.md                 # theory & teaching document
  COOKBOOK.md              # task-oriented recipes (all four languages)
  API_SPEC.md              # normative cross-language API contract
  docs/
    ARCHITECTURE.md        # design, data flow, numerical decisions
    diagrams/*.mmd         # raw Mermaid sources for the embedded diagrams
  data/
    portfolio.csv          # 10-position mixed equity/FX demo book
    spots_timeseries.csv   # 2y synthetic daily closes, 3 underliers (seeded)
    generate_data.py       # regenerates the data + golden values (Python only)
    golden/golden.json     # cross-language golden values (28 cases, full double precision)
  python/
    src/dpe/               # reference implementation (src layout)
    tests/                 # pytest suite incl. golden-value tests
    demo.py                # end-to-end demo (output excerpted below)
    requirements.txt
  cpp/
    include/dpe/*.hpp      # public headers, namespace dpe
    src/*.cpp              # implementation
    tests/*.cpp            # GoogleTest suite incl. golden values
    CMakeLists.txt         # C++17; lib + demo + tests + optional benchmark/profiling targets
    README.md              # C++ build/test/sanitizer/benchmark/profiling guide
    benchmarks/            # Google Benchmark microbenchmarks (optional)
    build.sh
  rust/
    Cargo.toml             # edition 2021; deps: rand, rand_distr, serde, serde_json
    src/                   # lib.rs + modules; Result<_, DpeError> error handling
    src/bin/demo.rs        # demo binary
    tests/                 # integration tests incl. golden values
  java/
    src/main/java/com/quant/dpe/    # Java 21, javadoc'd public API
    src/test/java/com/quant/dpe/    # JUnit 4 tests incl. golden values
    build.sh  test.sh  demo.sh
  .gitignore               # cpp/build, rust/target, java/out, __pycache__
```

## Build, test and run

### Python (3.11+, numpy + pandas; pytest for tests)

```bash
cd python
pip install -r requirements.txt      # numpy, pandas, pytest
PYTHONPATH=src python3 -m pytest -q  # run the test suite (81 functions / 150 collected cases)
PYTHONPATH=src python3 demo.py       # run the end-to-end demo
```

### C++ (C++17, CMake ≥ 3.16, GoogleTest via `find_package`)

```bash
cd cpp
bash build.sh                        # = cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
ctest --test-dir build --output-on-failure    # run the tests (61)
./build/demo                                  # run the demo (target name: demo)
```

The C++ port uses target-based CMake and GoogleTest/CTest. It also has opt-in ASan+UBSan builds, Google Benchmark microbenchmarks, profiling-ready builds for `perf`/Instruments, and strict compiler warnings. GitHub Actions runs the normal suite plus a sanitizer job and benchmark smoke test. See [`cpp/README.md`](cpp/README.md) for reproducible commands.

The implementation is intentionally C++17 and uses strong `enum class` types, `constexpr` constants, fixed-width integer types, value semantics/RAII, and the C++ standard library rather than C-style ownership.

Compiled with `-Wall -Wextra`; the build is warning-free with g++ 13.

### Rust (1.95, edition 2021)

```bash
cd rust
cargo build --release                # zero warnings
cargo test --release                 # 46 unit + integration tests + 1 doctest, incl. golden
cargo run --release --bin demo       # run the demo
```

### Java (21, JUnit 4 + Hamcrest at /usr/share/java/{junit4,hamcrest}.jar — no Maven/Gradle)

```bash
cd java
bash build.sh                        # javac -Xlint:all -Werror everything into out/
bash test.sh                         # JUnitCore on all *Test classes (54 tests)
bash demo.sh                         # run the demo main
```

The `.sh` files are committed with the executable bit, but `bash script.sh`
works regardless of how the clone preserved file modes.

## Demo output (Python reference, truncated)

```text
dpe — Derivatives Pricing Engine demo (Python reference implementation)

------------------------------------------------------------------------------
1. Analytic Black-Scholes-Merton and Garman-Kohlhagen
------------------------------------------------------------------------------
instrument                   price    delta    gamma     vega    theta      rho    vanna    volga
Equity ATM call (q=2%)      9.2270   0.5869   0.0190  37.9012  -5.0893  49.4581  -0.0948   2.3688
Equity ATM put  (q=2%)      6.3301  -0.3933   0.0190  37.9012  -2.2936 -45.6648  -0.0948   2.3688
High-vol small cap call     5.1945   0.6249   0.0263   6.8263  -2.9726   6.0532   0.2283  -1.3083
Weekly put (T=1/52)         0.5793  -0.2687   0.0951   4.5743 -28.9091  -0.5279  -0.7680   6.5677
LEAPS call (T=3y)          13.6862   0.4672   0.0077  65.0717  -3.4237  99.1088   0.6638   1.1193
FX EURUSD call (rd,rf)      0.0243   0.4365   5.0221   0.3038  -0.0345   0.2279   0.8573   0.0992
FX neg-rate put (rd<0)      0.0485  -0.4955   3.4958   0.3786  -0.0258  -0.5192   0.3653  -0.0035

FX delta conventions (EURUSD call): spot delta = 0.4365, forward delta = 0.4409, forward = 1.1055

------------------------------------------------------------------------------
2. Implied volatility round-trip
------------------------------------------------------------------------------
k= 120.0  price=   0.2519  sigma_in=0.1500  implied=0.1500000000
k=  80.0  price=  23.5592  sigma_in=0.4000  implied=0.4000000000
below-intrinsic price correctly rejected: price 1.0 violates the no-arbitrage
lower bound 21.191044831754994 (discounted intrinsic); no implied vol exists

------------------------------------------------------------------------------
3. Binomial trees (CRR / Jarrow-Rudd)
------------------------------------------------------------------------------
 steps   CRR-BS error   CRR+Richardson   JR-BS error
    50      -3.99e-02        -2.73e-03      3.69e-02
   200      -9.99e-03        -6.34e-04     -5.29e-03
   800      -2.50e-03        -1.55e-04     -3.90e-04
  2000      -1.00e-03        -6.19e-05      7.62e-04

American put early-exercise premium (s=100, T=1, sigma=20%, r=5%):
  strike    european    american   premium
    90.0      2.3094      2.4724    0.1630
   100.0      5.5695      6.0888    0.5193
   110.0     10.6775     11.9744    1.2969
   120.0     17.3941     20.1358    2.7417

------------------------------------------------------------------------------
4. Monte Carlo (100k paths, seed 42)
------------------------------------------------------------------------------
European call, BS analytic = 10.4506
estimator                        price   std err                  95% CI
plain                          10.4205    0.0468      [10.3289, 10.5122]
antithetic                     10.4673    0.0331      [10.4025, 10.5321]
antithetic + control           10.4409    0.0088      [10.4237, 10.4582]

exotic                                 price   std err                  95% CI
arithmetic Asian (12 fix, CV)         6.1569    0.0008        [6.1554, 6.1585]
up-and-out barrier B=130              3.6940    0.0171        [3.6604, 3.7276]
floating-strike lookback             16.2816    0.0266      [16.2295, 16.3338]

delta: analytic 0.6368, pathwise 0.6379 (se 0.0006), FD+CRN 0.6378 (se 0.0006)

------------------------------------------------------------------------------
6. Portfolio risk report (data/portfolio.csv)
------------------------------------------------------------------------------
id              typ style     price    delta    gamma      vega     theta  position value
EQ_DIV_C1       C   euro    11.2248   0.6565   0.0179   32.5800   -5.7081        1,122.48
...
FX_NEGRATE_P1   P   euro     0.0485  -0.4955   3.4958    0.3786   -0.0258       36,341.93

total book value                                                55,987.78
```

## Golden values — how cross-language consistency is enforced

`data/golden/golden.json` holds 28 named reference cases in a deliberately
flat, easy-to-parse schema:

```json
{ "cases": [ { "name": "bs_call_atm",
               "inputs": { "s": 100.0, "k": 100.0, "t": 1.0, "sigma": 0.2,
                           "r": 0.05, "q": 0.0, "type": "call" },
               "expect": { "price": 10.450583572185565 },
               "tol": 1e-8 } ] }
```

* The file is **generated once** by the Python reference
  (`python3 data/generate_data.py`, fixed seed 20260827), after
  self-validation against analytic anchors, put-call parity, implied-vol
  round-trips, tree convergence and a Monte Carlo 3-standard-error check.
  Ports never regenerate it; they only consume it.
* **Every language's test suite loads the same JSON file** and asserts each
  case within its absolute tolerance `tol` (C++ with a minimal hand-rolled
  reader for the flat schema, Rust with `serde_json`, Java with a small
  bundled parser class). A port that drifts from the reference — a wrong
  sign in theta, a `d2` slip, a mis-scaled vega — fails its own build.
* Expected values are written at **full double precision** (`json.dump`
  round-trips doubles exactly), so the whole of `tol` is available to the
  port — an earlier revision rounded to 10 significant digits, which for a
  value like `rho = 53.23…` silently consumed half of a `1e-8` budget.
* Deterministic cases (analytic, trees, implied vol) carry tight tolerances
  (`1e-8`, and `1e-10` for the deep-OTM tail-precision case). The five Monte
  Carlo cases use statistical tolerances (0.15 / 0.05 for the vanilla and
  Asian at 100k paths; 0.06 / 0.10 / 0.07 for the single-monitoring-date
  barrier and lookback cases at 200k paths, whose expected values are
  closed-form: `C(k) − C(B) − (B − k)e^{−rt}N(d2(B))` and the ATM vanilla)
  because RNG streams are language-specific; each port uses its own fixed
  seed and stays deterministic within itself.
* No suite pins the exact case count (all assert `>= 20`), and every suite
  fails on a case name it cannot dispatch, so a new case breaks all four
  builds until all four implement it.

Beyond the golden suite, each language's tests independently verify put-call
parity over a grid (analytic, and exactly on the CRR lattice), Greeks
against central finite differences, price monotonicity in `s` and `k`,
American ≥ European (and the dividend-driven early-exercise premium of a
call), barrier ≤ vanilla, Asian ≤ vanilla, the one-step CRR closed form,
the exact `t = 0` / `sigma = 0` limits and the at-the-money tie-break, the
`steps = 2` tree Greeks, implied-vol non-convergence reporting and
`tol`/`max_iter` validation, the seed and `rel_bump` domains, the lattice
overflow guard, bit-identity of the single-fixing Asian and the vanilla
engine, and the σ = 0 Monte Carlo limit. Final counts: Python 81 test
functions / 150 collected cases, C++ 61, Rust 47 (46 + doctest), Java 54.

## Real-world usage notes

What a practitioner gets, precisely:

* **Units and conventions.** `t` is a year fraction (the library does no
  day-count or calendar work; the demo book's `t` values were produced with
  ACT/365F); `sigma`, `r`, `q` are decimals with continuous compounding;
  `theta` is per calendar **year** (divide by 365 yourself), `vega`/`vanna`/
  `volga` per unit of vol (divide by 100 for per vol point), `rho` per unit
  of rate. FX uses `r = rd`, `q = rf`, spot quoted domestic-per-foreign,
  premium in domestic currency per unit of foreign notional; `gk_greeks`
  reports the premium-excluded **spot** delta (forward delta via
  `spot_delta_to_forward`). No premium-included deltas, no dividend/foreign
  rho, no strike-from-delta solver.
* **What is validated at every public entry point** (identically in all four
  ports, see `API_SPEC.md` §1, §5, §8): `s > 0`, `k >= 0`, `t >= 0`,
  `sigma >= 0`, `r`/`q` finite, NaN/inf anywhere rejected; enum strings
  case-insensitive; `steps >= 1` (`>= 2` for tree Greeks), `n_paths >= 2`
  (`>= 4` and even with antithetics), `n_steps >= 1`, `n_fixings >= 1`,
  `seed` in `[0, 2^63 − 1]`, `rel_bump` in `(0, 1)`, `tol > 0`,
  `max_iter >= 1`, `barrier > 0`, `periods_per_year >= 1`. Python accepts
  any `numbers.Integral` (numpy/pandas ints) and rejects bools and
  integral floats.
* **What can still fail loudly, by design.** Implied vol raises on a price
  at/outside the no-arbitrage bounds (a deep-ITM screen price sitting
  *exactly* at discounted intrinsic is rejected, not returned as σ = 0),
  on a price indistinguishable from the upper bound (vol cap 20.0), and
  when `max_iter` is exhausted before `|model − price| < tol` — never a
  half-converged root. Trees raise when the risk-neutral probability leaves
  `(0, 1)` at the chosen step size and when the terminal nodes would
  overflow (`|ln s| + |(r − q − σ²/2)t| + σ√(t·steps) > 700`).
* **Parameter bounds that are your responsibility.** Rates and yields are
  only required to be finite; `|r·t|`, `|q·t|` beyond ~700 overflow the
  discount factors (an absurd regime that is not guarded). The implied-vol
  tolerance is **absolute in price** (`tol = 1e-10` by default): for very
  low-premium wings or high-notional quotes scale the price or loosen `tol`
  accordingly — the bracket-width rule (`< 1e-12` in vol) still terminates
  the solve. Expiring (`t = 0`) exactly-ATM options report the one-sided
  delta (`1`/`0`), not `0.5` (`API_SPEC.md` §3.2).
* **Monte Carlo.** Exact GBM stepping (no Euler bias in marginals); barrier
  and lookback are the **discretely monitored** contracts with documented
  O(1/√n_steps) bias and no Broadie–Glasserman–Kou correction; standard
  errors are honest (pair averages under antithetics, in-sample control
  β). Determinism holds per language for a fixed toolchain (`API_SPEC.md`
  §5); streams differ across languages. Python and Java materialise the
  `n_paths × n_steps` path matrix (≈ 16 bytes per path-step); C++ and Rust
  stream paths in O(n_steps) memory.
* **Deliberately out of scope.** Day-count conventions, holiday calendars,
  settlement lags, discrete cash dividends, term structures of rates or vol
  (all inputs are flat scalars), American exercise outside the lattice,
  premium-included FX deltas, digital/discontinuous payoffs in the pathwise
  delta.
* **Verification status.** Every formula has been checked line by line
  against the references below and cross-checked across the four ports by
  the golden suite; the test suites cover the analytic identities, limits
  and error contracts listed above. This is a verified reference
  implementation of textbook models, not a production risk system: it has
  no market-data, calendar or persistence layer and its models are the
  flat-parameter Black–Scholes–Merton world.

## References

Works the code implements or the documentation relies on:

1. F. Black and M. Scholes (1973), "The Pricing of Options and Corporate
   Liabilities", *Journal of Political Economy* 81(3), 637–654.
   https://doi.org/10.1086/260062
2. R. C. Merton (1973), "Theory of Rational Option Pricing", *Bell Journal
   of Economics and Management Science* 4(1), 141–183.
   https://doi.org/10.2307/3003143
3. M. B. Garman and S. W. Kohlhagen (1983), "Foreign Currency Option
   Values", *Journal of International Money and Finance* 2(3), 231–237.
   https://doi.org/10.1016/S0261-5606(83)80001-1
4. J. C. Cox, S. A. Ross and M. Rubinstein (1979), "Option Pricing: A
   Simplified Approach", *Journal of Financial Economics* 7(3), 229–263.
   https://doi.org/10.1016/0304-405X(79)90015-1
5. R. A. Jarrow and A. Rudd (1983), *Option Pricing*, Richard D. Irwin,
   Homewood IL (ch. 13, the equal-probability lattice).
6. A. G. Z. Kemna and A. C. F. Vorst (1990), "A Pricing Method for Options
   Based on Average Asset Values", *Journal of Banking & Finance* 14(1),
   113–129. https://doi.org/10.1016/0378-4266(90)90039-5
7. M. Broadie, P. Glasserman and S. Kou (1997), "A Continuity Correction
   for Discrete Barrier Options", *Mathematical Finance* 7(4), 325–349.
   https://doi.org/10.1111/1467-9965.00035 (documented, not applied)
8. M. Broadie and P. Glasserman (1996), "Estimating Security Price
   Derivatives Using Simulation", *Management Science* 42(2), 269–285.
   https://doi.org/10.1287/mnsc.42.2.269 (pathwise delta)
9. P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
   Springer, New York (§4.1–4.2 control/antithetic variates, §7.1–7.2
   finite-difference and pathwise Greeks). https://doi.org/10.1007/978-0-387-21617-1
10. W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery
    (2007), *Numerical Recipes: The Art of Scientific Computing*, 3rd ed.,
    Cambridge University Press (§9.4, safeguarded Newton `rtsafe`).
11. W. J. Cody (1969), "Rational Chebyshev Approximations for the Error
    Function", *Mathematics of Computation* 23(107), 631–637.
    https://doi.org/10.1090/S0025-5718-1969-0247736-4
12. W. J. Cody (1993), "Algorithm 715: SPECFUN — A Portable FORTRAN Package
    of Special Function Routines and Test Drivers", *ACM Transactions on
    Mathematical Software* 19(1), 22–32. https://doi.org/10.1145/151271.151273
    (the `erfc` used by the Rust and Java ports)
13. E. G. Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd
    ed., McGraw-Hill, New York (§1.1–1.3 Greeks incl. vanna/volga, §4.20
    discrete geometric Asian, §7.1 lattice parameterisations).
14. J. C. Hull (2022), *Options, Futures, and Other Derivatives*, 11th ed.,
    Pearson (§15.4 realised volatility, §21.4 tree Greeks; the 10.4506 ATM
    anchor).
15. U. Wystup (2006), *FX Options and Structured Products*, Wiley,
    Chichester (ch. 1, spot/forward delta conventions).

## License

MIT — see [`LICENSE`](LICENSE). No warranty; not investment advice.