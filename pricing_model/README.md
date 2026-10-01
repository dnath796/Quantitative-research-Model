# Derivatives Pricing Engine

A cross-language derivatives pricing library implemented in **Python, C++, Rust, and Java**.

It covers vanilla options and first-generation exotics using three classic pricing approaches:

- **Black–Scholes–Merton / Garman–Kohlhagen**
- **Binomial trees**
- **Monte Carlo simulation**

The Python package is the reference implementation. C++, Rust, and Java implement the same public contract and are checked against a shared cross-language golden-value test suite.

## What it supports

### Analytic pricing

Black–Scholes–Merton for equities with continuous dividend yield and Garman–Kohlhagen for FX.

Includes:

- call and put prices
- delta
- gamma
- vega
- theta
- rho
- vanna
- volga
- implied volatility via safeguarded Newton iteration with bisection fallback

FX helpers include forward pricing and spot/forward delta conversion.

### Binomial trees

Two lattice models are available:

- Cox–Ross–Rubinstein
- Jarrow–Rudd

Supported features:

- European exercise
- American exercise
- tree delta, gamma, and theta
- optional two-point Richardson averaging

### Monte Carlo

Exact-step geometric Brownian motion simulation for:

- European vanilla options
- arithmetic Asian options
- up-and-out barriers
- floating-strike lookbacks

Variance reduction and risk estimation include:

- antithetic variates
- control variates
- standard errors
- 95% confidence intervals
- pathwise delta
- finite-difference delta with common random numbers

## Four languages, one contract

| Capability                             | Python | C++ | Rust | Java |
| -------------------------------------- | -----: | --: | ---: | ---: |
| BSM pricing with dividend yield        |      ✓ |   ✓ |    ✓ |    ✓ |
| Garman–Kohlhagen FX                    |      ✓ |   ✓ |    ✓ |    ✓ |
| Full Greeks incl. vanna/volga          |      ✓ |   ✓ |    ✓ |    ✓ |
| Implied volatility                     |      ✓ |   ✓ |    ✓ |    ✓ |
| CRR and Jarrow–Rudd trees              |      ✓ |   ✓ |    ✓ |    ✓ |
| European and American exercise         |      ✓ |   ✓ |    ✓ |    ✓ |
| Tree Greeks                            |      ✓ |   ✓ |    ✓ |    ✓ |
| Richardson averaging                   |      ✓ |   ✓ |    ✓ |    ✓ |
| Monte Carlo vanilla                    |      ✓ |   ✓ |    ✓ |    ✓ |
| Arithmetic Asian                       |      ✓ |   ✓ |    ✓ |    ✓ |
| Up-and-out barrier                     |      ✓ |   ✓ |    ✓ |    ✓ |
| Floating-strike lookback               |      ✓ |   ✓ |    ✓ |    ✓ |
| Pathwise / CRN finite-difference delta |      ✓ |   ✓ |    ✓ |    ✓ |
| FX helpers                             |      ✓ |   ✓ |    ✓ |    ✓ |
| Historical volatility                  |      ✓ |   ✓ |    ✓ |    ✓ |
| Shared golden-value tests              |      ✓ |   ✓ |    ✓ |    ✓ |
| Negative-rate support                  |      ✓ |   ✓ |    ✓ |    ✓ |

Python additionally accepts NumPy and pandas integer types for count and seed arguments.

## Quick example

Python:

```python
from dpe import bsm_price

price = bsm_price(
    s=100.0,
    k=100.0,
    t=1.0,
    sigma=0.20,
    r=0.05,
    q=0.00,
    option_type="call",
)

print(price)
# 10.450583572...
```

The same model behavior is implemented in C++, Rust, and Java under their respective public APIs.

## Repository layout

```text
derivatives-pricing-engine/
├── README.md
├── API_SPEC.md
├── LEARN.md
├── COOKBOOK.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GITHUB_PAGES.md
│   └── diagrams/
├── data/
│   ├── portfolio.csv
│   ├── spots_timeseries.csv
│   ├── generate_data.py
│   └── golden/golden.json
├── python/
│   ├── src/dpe/
│   ├── tests/
│   └── demo.py
├── cpp/
│   ├── include/dpe/
│   ├── src/
│   ├── tests/
│   └── CMakeLists.txt
├── rust/
│   ├── src/
│   └── tests/
└── java/
    ├── src/main/java/com/quant/dpe/
    └── src/test/java/com/quant/dpe/
```

## Documentation

| File                   | Purpose                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| `API_SPEC.md`          | Normative cross-language API contract, units, errors, and edge cases                           |
| `LEARN.md`             | Theory and derivations for BSM/GK, Greeks, implied vol, trees, Monte Carlo, and FX conventions |
| `COOKBOOK.md`          | Task-oriented examples in all four languages                                                   |
| `docs/ARCHITECTURE.md` | Component design, numerical decisions, data flow, and testing strategy                         |
| `docs/GITHUB_PAGES.md` | Documentation-site publishing instructions                                                     |

## Build and test

### Python

Requires Python 3.11+, NumPy, pandas, and pytest.

```bash
cd python
pip install -r requirements.txt

PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 demo.py
```

The Python suite contains 81 test functions and 150 collected cases.

### C++

Requires C++17, CMake 3.16+, and GoogleTest.

```bash
cd cpp

bash build.sh
ctest --test-dir build --output-on-failure
./build/demo
```

The project builds with `-Wall -Wextra` and is warning-free with g++ 13.

### Rust

Rust edition 2021.

```bash
cd rust

cargo build --release
cargo test --release
cargo run --release --bin demo
```

The suite contains 46 unit/integration tests plus one doctest.

### Java

Requires Java 21, JUnit 4, and Hamcrest.

```bash
cd java

bash build.sh
bash test.sh
bash demo.sh
```

The Java build uses `javac -Xlint:all -Werror`.

## Example results

For a standard ATM European call:

```text
S = 100
K = 100
T = 1
sigma = 20%
r = 5%
q = 0
```

the analytic Black–Scholes price is approximately:

```text
10.4506
```

A Monte Carlo run with 100,000 paths produces estimates consistent with the analytic value, with progressively smaller standard errors when antithetic and control variates are enabled.

The demo also covers:

- implied-volatility round trips
- CRR and Jarrow–Rudd convergence
- American put early-exercise premia
- arithmetic Asian options
- barrier options
- lookbacks
- pathwise and finite-difference deltas
- portfolio-level pricing and Greeks

## Cross-language verification

All four implementations consume the same golden-value file:

```text
data/golden/golden.json
```

The reference values are generated from Python and stored at full double precision.

Each language loads the same cases and checks its results against the case-specific tolerance.

That means implementation drift—such as:

- a wrong theta sign
- an incorrect `d2`
- a mis-scaled vega
- inconsistent edge-case handling

causes the affected language’s test suite to fail.

The golden suite contains deterministic analytic, tree, and implied-volatility cases as well as Monte Carlo cases with statistical tolerances.

Ports never regenerate the reference values themselves.

## Testing beyond golden values

Each language also verifies core numerical identities and invariants independently.

Tests include:

- put–call parity
- Greeks against central finite differences
- monotonicity in spot and strike
- American value ≥ European value
- dividend-driven early exercise
- barrier value ≤ vanilla value
- Asian value ≤ vanilla value
- one-step CRR closed form
- exact `t = 0` limits
- exact `sigma = 0` limits
- implied-volatility round trips
- invalid input handling
- non-convergence reporting
- lattice overflow protection
- deterministic seeded Monte Carlo behavior

## Numerical conventions

Inputs use the following conventions:

- `t`: year fraction
- `sigma`: annualized volatility as a decimal
- `r`: continuously compounded rate
- `q`: continuously compounded dividend yield or foreign rate
- `theta`: per calendar year
- `vega`, `vanna`, `volga`: per unit volatility
- `rho`: per unit rate

For example:

- divide theta by 365 for an approximate daily value
- divide vega by 100 for sensitivity to one volatility point

For FX:

```text
r = domestic rate
q = foreign rate
```

Spot is quoted as domestic currency per unit of foreign currency.

The library reports premium-excluded spot delta. Forward delta can be obtained through the provided conversion helper.

## Input validation

Public entry points consistently validate inputs across all four languages.

Among other rules:

```text
s > 0
k >= 0
t >= 0
sigma >= 0
steps >= 1
n_paths >= 2
n_steps >= 1
n_fixings >= 1
barrier > 0
tol > 0
max_iter >= 1
seed ∈ [0, 2^63 - 1]
rel_bump ∈ (0, 1)
```

NaN and infinity are rejected.

Tree Greeks require at least two steps.

Antithetic Monte Carlo requires an even path count of at least four.

## Implied volatility

The implied-volatility solver uses a bracketed Newton method with bisection safeguards.

It fails explicitly when:

- the option price violates no-arbitrage bounds
- the price is effectively at the solver’s upper-volatility limit
- convergence is not achieved before `max_iter`

It never returns a partially converged root as if it were valid.

The default tolerance is absolute in price rather than relative.

## Monte Carlo implementation

Simulation uses exact geometric Brownian motion stepping rather than Euler discretization.

Antithetic samples are handled as pair averages when calculating standard errors.

Control-variate coefficients are estimated in-sample.

Random streams are deterministic within each language for a fixed toolchain and seed, but streams are not expected to be identical across languages.

Memory behavior differs by implementation:

- Python and Java materialize the path matrix
- C++ and Rust stream paths using approximately `O(n_steps)` memory

Barrier and lookback options are discretely monitored.

No Broadie–Glasserman–Kou continuity correction is applied.

## Deliberate scope limits

This project intentionally does not implement:

- holiday calendars
- day-count engines
- settlement lags
- discrete cash dividends
- interest-rate term structures
- volatility surfaces
- market-data feeds
- persistence
- American exercise outside the lattice
- premium-included FX delta
- strike-from-delta inversion
- digital/discontinuous-payoff pathwise delta

The models assume flat scalar parameters in the Black–Scholes–Merton framework.

## Project goal

This is a **verified reference implementation of textbook derivatives models**, not a production trading or risk platform.

The emphasis is on:

- transparent formulas
- reproducible numerical behavior
- explicit edge cases
- cross-language consistency
- strong automated tests
- readable implementations suitable for learning, validation, and experimentation

## References

The implementation and documentation draw on the standard literature, including:

1. Black & Scholes (1973), *The Pricing of Options and Corporate Liabilities*
2. Merton (1973), *Theory of Rational Option Pricing*
3. Garman & Kohlhagen (1983), *Foreign Currency Option Values*
4. Cox, Ross & Rubinstein (1979), *Option Pricing: A Simplified Approach*
5. Jarrow & Rudd (1983), *Option Pricing*
6. Kemna & Vorst (1990), arithmetic/geometric Asian-option methods
7. Broadie, Glasserman & Kou (1997), discrete barrier continuity correction
8. Broadie & Glasserman (1996), simulation-based derivative estimators
9. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*
10. Press et al. (2007), *Numerical Recipes*
11. Cody (1969, 1993), numerical special-function approximations
12. Haug (2007), *The Complete Guide to Option Pricing Formulas*
13. Hull (2022), *Options, Futures, and Other Derivatives*
14. Wystup (2006), *FX Options and Structured Products*

## License

MIT License. See `LICENSE`. No warranty. Not investment advice.
