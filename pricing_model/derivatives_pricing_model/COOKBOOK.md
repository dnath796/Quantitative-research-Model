# COOKBOOK — Practical Recipes

Copy-paste examples for common derivatives-pricing tasks in **Python, C++, Rust, and Java**.

Every snippet in this cookbook is tested against the real library as part of the release process. The Python examples run directly; the C++, Rust, and Java examples are compiled and executed against their corresponding implementations.

If you want API contracts, validation rules, and edge cases, see `API_SPEC.md`. For derivations and theory, see `LEARN.md`.

## Conventions

Across the library:

- `t` is measured in years
- `sigma`, `r`, and `q` are decimals
- rates and yields use continuous compounding
- theta is reported per year
- vega, vanna, and volga are reported per unit volatility
- FX uses `r = rd` and `q = rf`
- FX spot is quoted domestic currency per unit of foreign currency

The Python API accepts validated strings while the native APIs use enums.

| Concept | Python | C++ | Rust | Java |
|---|---|---|---|---|
| Call | `"call"` | `OptionType::Call` | `OptionType::Call` | `OptionType.CALL` |
| Put | `"put"` | `OptionType::Put` | `OptionType::Put` | `OptionType.PUT` |
| European | `"european"` | `ExerciseStyle::European` | `ExerciseStyle::European` | `ExerciseStyle.EUROPEAN` |
| American | `"american"` | `ExerciseStyle::American` | `ExerciseStyle::American` | `ExerciseStyle.AMERICAN` |
| CRR | `"crr"` | `TreeMethod::CRR` | `TreeMethod::Crr` | `BinomialMethod.CRR` |
| Jarrow–Rudd | `"jr"` | `TreeMethod::JR` | `TreeMethod::Jr` | `BinomialMethod.JR` |

Strings read from files can be converted with:

- C++: `dpe::parse_option_type(s)`
- Rust: `s.parse::<dpe::OptionType>()?`
- Java: `OptionType.parse(s)`

Parsing is case-insensitive. Invalid values are rejected.

---

## 1. Price a European option and calculate the Greeks

For a dividend-paying equity option, use `bs_greeks`.

### Python

```python
import dpe

g = dpe.bs_greeks(
    s=105.0,
    k=100.0,
    t=0.75,
    sigma=0.22,
    r=0.04,
    q=0.02,
    option_type="call",
)

print(f"price {g.price:.6f}")
print(f"delta {g.delta:.4f}  gamma {g.gamma:.4f}")
print(f"vega {g.vega:.4f}  theta/yr {g.theta:.4f}  rho {g.rho:.4f}")
print(f"vanna {g.vanna:.4f}  volga {g.volga:.4f}")

print(f"theta/day {g.theta / 365:.4f}")
print(f"vega/volpt {g.vega / 100:.4f}")
```

### C++

```cpp
#include "dpe/dpe.hpp"
#include <cstdio>

int main() {
    dpe::Greeks g = dpe::bs_greeks(
        105.0, 100.0, 0.75, 0.22, 0.04, 0.02,
        dpe::OptionType::Call
    );

    std::printf(
        "price %.6f delta %.4f gamma %.4f vega %.4f theta %.4f\n",
        g.price, g.delta, g.gamma, g.vega, g.theta
    );

    std::printf(
        "rho %.4f vanna %.4f volga %.4f\n",
        g.rho, g.vanna, g.volga
    );
}
```

### Rust

```rust
use dpe::OptionType;

fn main() -> Result<(), dpe::DpeError> {
    let g = dpe::bs_greeks(
        105.0, 100.0, 0.75, 0.22, 0.04, 0.02,
        OptionType::Call,
    )?;

    println!(
        "price {:.6} delta {:.4} gamma {:.4}",
        g.price, g.delta, g.gamma
    );

    println!(
        "vega {:.4} theta {:.4} rho {:.4} vanna {:.4} volga {:.4}",
        g.vega, g.theta, g.rho, g.vanna, g.volga
    );

    Ok(())
}
```

### Java

```java
import com.quant.dpe.BlackScholes;
import com.quant.dpe.Greeks;
import com.quant.dpe.OptionType;

public class Recipe1 {
    public static void main(String[] args) {
        Greeks g = BlackScholes.bsGreeks(
                105.0, 100.0, 0.75, 0.22, 0.04, 0.02,
                OptionType.CALL);

        System.out.printf(
                "price %.6f delta %.4f gamma %.4f%n",
                g.price(), g.delta(), g.gamma());

        System.out.printf(
                "vega %.4f theta %.4f rho %.4f vanna %.4f volga %.4f%n",
                g.vega(), g.theta(), g.rho(), g.vanna(), g.volga());
    }
}
```

---

## 2. Price an FX option from a volatility quote

Suppose EURUSD spot is `1.10`, the six-month `1.12` call is quoted at `10%` vol, USD rates are `3%`, and EUR rates are `2%`.

Garman–Kohlhagen is Black–Scholes with:

```text
r = domestic rate
q = foreign rate
```

### Python

```python
import dpe

s, k, t = 1.10, 1.12, 0.5
vol, rd, rf = 0.10, 0.03, 0.02

px = dpe.gk_price(s, k, t, vol, rd, rf, "call")
g = dpe.gk_greeks(s, k, t, vol, rd, rf, "call")

notional_eur = 1_000_000

print(f"premium {px:.6f} USD per EUR")
print(f"premium {px * 1e4:.1f} pips")
print(f"cash premium ${px * notional_eur:,.2f}")
print(f"spot delta {g.delta:.4f}")
print(f"forward {dpe.fx_forward(s, t, rd, rf):.4f}")
```

The returned premium is in domestic currency per unit of foreign notional.

---

## 3. Convert FX spot delta to forward delta

`gk_greeks(...).delta` returns a premium-excluded **spot delta**.

A forward delta differs only by the foreign discount factor.

### Python

```python
import dpe

t, rf = 0.5, 0.02

d_spot = dpe.gk_greeks(
    1.10, 1.12, t, 0.10, 0.03, rf, "call"
).delta

d_fwd = dpe.spot_delta_to_forward(d_spot, t, rf)
back = dpe.forward_delta_to_spot(d_fwd, t, rf)

print(f"spot {d_spot:.6f}")
print(f"forward {d_fwd:.6f}")
print(f"round trip {back:.6f}")
```

The conversion preserves sign, so it works for puts as well.

---

## 4. Recover implied volatility and reject invalid prices

The implied-volatility solver checks no-arbitrage bounds before iteration.

It also validates `tol` and `max_iter`. If convergence is not achieved, it raises an error rather than returning a partially converged value.

### Python

```python
import dpe

iv = dpe.implied_vol(
    price=10.45058357,
    s=100.0,
    k=100.0,
    t=1.0,
    r=0.05,
    q=0.0,
    option_type="call",
)

print(f"implied vol {iv:.10f}")
```

Invalid prices fail explicitly:

```python
try:
    dpe.implied_vol(
        price=1.0,
        s=100.0,
        k=80.0,
        t=1.0,
        r=0.05,
        q=0.0,
        option_type="call",
    )
except ValueError as e:
    print(f"rejected: {e}")
```

A deliberately insufficient iteration budget also fails:

```python
try:
    dpe.implied_vol(
        10.45058357,
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        "call",
        tol=1e-10,
        max_iter=1,
    )
except ValueError as e:
    print(f"not converged: {e}")
```

When building a volatility surface, handle failures quote-by-quote rather than silently accepting them.

---

## 5. Price an American option and measure its early-exercise premium

Price the American contract with a CRR tree and compare it with its European counterpart.

### Python

```python
import dpe

s, t, sigma, r, q = 100.0, 1.0, 0.20, 0.05, 0.0

for k in (90.0, 100.0, 110.0, 120.0):
    amer = dpe.binomial_price(
        s, k, t, sigma, r, q,
        "put", "american", 500, "crr"
    )

    euro = dpe.bs_price(
        s, k, t, sigma, r, q, "put"
    )

    print(
        f"k={k:6.1f}  "
        f"american {amer:8.4f}  "
        f"european {euro:8.4f}  "
        f"premium {amer - euro:7.4f}"
    )
```

For these puts, the early-exercise premium is positive and increases as the option moves further in the money.

---

## 6. Improve tree accuracy with Richardson averaging

Binomial prices often oscillate between odd and even step counts.

Setting `richardson=True` averages the `steps` and `steps + 1` prices and typically improves convergence at roughly twice the computational cost.

### Python

```python
import dpe

bs = dpe.bs_price(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call"
)

for n in (50, 200, 800):
    plain = dpe.binomial_price(
        100, 100, 1.0, 0.2, 0.05, 0.0,
        "call", "european", n, "crr"
    )

    rich = dpe.binomial_price(
        100, 100, 1.0, 0.2, 0.05, 0.0,
        "call", "european", n, "crr",
        richardson=True,
    )

    print(
        f"n={n:4d}  "
        f"plain error {plain - bs:+.2e}  "
        f"richardson error {rich - bs:+.2e}"
    )
```

---

## 7. Price with Monte Carlo and interpret the error bar

Every Monte Carlo routine returns an `MCResult` containing:

```text
value
std_error
ci_low
ci_high
n_paths
```

For a European option, the analytic Black–Scholes value gives a convenient benchmark.

### Python

```python
import dpe

bs = dpe.bs_price(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call"
)

plain = dpe.mc_european(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call",
    n_paths=100_000,
    seed=42,
    antithetic=False,
)

cv = dpe.mc_european(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call",
    n_paths=100_000,
    seed=42,
    antithetic=True,
    control_variate=True,
)

print(f"analytic {bs:.4f}")

print(
    f"plain {plain.value:.4f} ± {plain.std_error:.4f}  "
    f"CI [{plain.ci_low:.4f}, {plain.ci_high:.4f}]"
)

print(
    f"anti+cv {cv.value:.4f} ± {cv.std_error:.4f}  "
    f"CI [{cv.ci_low:.4f}, {cv.ci_high:.4f}]"
)
```

The control variate uses discounted terminal spot, whose expectation is known, and substantially reduces estimator variance at the same path count.

---

## 8. Price an arithmetic Asian with a control variate

The arithmetic Asian engine can use a geometric-Asian control variate.

Because the geometric average has a closed-form price and is highly correlated with the arithmetic payoff, this can reduce the Monte Carlo standard error dramatically.

### Python

```python
import dpe

geo = dpe.geometric_asian_price(
    100, 100, 1.0, 0.2, 0.05, 0.0,
    "call",
    n_fixings=12,
)

cv = dpe.mc_asian_arithmetic(
    100, 100, 1.0, 0.2, 0.05, 0.0,
    "call",
    n_paths=100_000,
    n_steps=12,
    seed=42,
    antithetic=True,
    control_variate=True,
)

raw = dpe.mc_asian_arithmetic(
    100, 100, 1.0, 0.2, 0.05, 0.0,
    "call",
    n_paths=100_000,
    n_steps=12,
    seed=42,
    antithetic=True,
    control_variate=False,
)

print(f"geometric closed form {geo:.4f}")
print(f"arithmetic no CV     {raw.value:.4f} ± {raw.std_error:.4f}")
print(f"arithmetic with CV   {cv.value:.4f} ± {cv.std_error:.4f}")
```

The arithmetic average is at least the geometric average path-by-path by AM–GM.

---

## 9. Price an up-and-out barrier and inspect monitoring bias

The barrier engine prices a **discretely monitored** contract.

Monitoring includes `t = 0`, so an option with `s >= barrier` is immediately knocked out.

### Python

```python
import dpe

vanilla = dpe.bs_price(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call"
)

for n_steps in (25, 100, 400):
    r = dpe.mc_barrier_up_out(
        100, 100, 1.0, 0.2, 0.05, 0.0,
        "call",
        barrier=130.0,
        n_paths=100_000,
        n_steps=n_steps,
        seed=42,
    )

    print(
        f"n_steps={n_steps:4d}  "
        f"price {r.value:.4f} ± {r.std_error:.4f}"
    )

print(f"vanilla upper bound {vanilla:.4f}")
```

Relative to a continuously monitored barrier, discrete monitoring produces an upward bias of order:

```text
O(1 / sqrt(n_steps))
```

Increasing the monitoring frequency exposes more barrier crossings, so the discrete price falls toward the continuous-monitoring value.

The library does not automatically apply a Broadie–Glasserman–Kou continuity correction.

---

## 10. Estimate Monte Carlo delta without excessive noise

Two estimators are provided.

### Pathwise delta

Differentiate the payoff along each simulated path.

It generally has low variance but requires a sufficiently smooth payoff.

### Finite-difference delta with common random numbers

Reprice at:

```text
S(1 - h)
S(1 + h)
```

using the same random draws for both simulations.

Do not estimate the difference using two independent Monte Carlo runs.

### Python

```python
import dpe

analytic = dpe.bs_greeks(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call"
).delta

pw = dpe.mc_delta_pathwise(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call",
    n_paths=100_000,
    seed=42,
)

fd = dpe.mc_delta_fd_crn(
    100, 100, 1.0, 0.2, 0.05, 0.0, "call",
    n_paths=100_000,
    seed=42,
    rel_bump=1e-4,
)

print(f"analytic {analytic:.4f}")
print(f"pathwise {pw.value:.4f} ± {pw.std_error:.4f}")
print(f"FD+CRN   {fd.value:.4f} ± {fd.std_error:.4f}")
```

---

## 11. Estimate historical volatility from a price series

`historical_vol` computes close-to-close realized volatility as:

```text
sample standard deviation of log returns × sqrt(periods_per_year)
```

For daily data, use `252` trading periods per year.

### Python

```python
import csv
import dpe

with open("../data/spots_timeseries.csv") as f:
    rows = list(csv.DictReader(f))

for name in ("ACME", "GLOBEX", "PIPCO"):
    prices = [float(r[name]) for r in rows]

    vol = dpe.historical_vol(
        prices,
        periods_per_year=252,
    )

    print(
        f"{name:8s} "
        f"last {prices[-1]:10.4f} "
        f"realised vol {vol:6.2%}"
    )
```

Comparing realized and implied volatility can be useful when studying differences between historical and option-implied variance.

---

## 12. Build a small portfolio risk report

The sample portfolio in:

```text
data/portfolio.csv
```

contains mixed equity and FX positions, including one American option.

Use:

- analytic Greeks for European rows
- a 500-step CRR tree for American rows
- `r = rd`, `q = rf` for FX rows

### Python

```python
import pandas as pd
import dpe

book = pd.read_csv("../data/portfolio.csv")

rows = []
total = 0.0

for p in book.itertuples():
    if p.style == "american":
        g = dpe.binomial_greeks(
            p.s, p.k, p.t, p.sigma, p.r, p.q,
            p.type,
            "american",
            500,
            "crr",
        )
    else:
        g = dpe.bs_greeks(
            p.s, p.k, p.t, p.sigma, p.r, p.q,
            p.type,
        )

    value = g.price * p.quantity
    total += value

    rows.append(
        (
            p.id,
            g.price,
            g.delta * p.quantity,
            g.gamma * p.quantity,
            value,
        )
    )

print(
    f"{'id':16s}"
    f"{'price':>10s}"
    f"{'pos delta':>14s}"
    f"{'pos gamma':>14s}"
    f"{'value':>15s}"
)

for rid, price, delta, gamma, value in rows:
    print(
        f"{rid:16s}"
        f"{price:10.4f}"
        f"{delta:14,.2f}"
        f"{gamma:14,.2f}"
        f"{value:15,.2f}"
    )

print(
    f"{'total book value':>54s}"
    f"{total:15,.2f}"
)
```

Tree Greeks do not include vega.

---

## Building the native examples

From the repository root, first build the corresponding library.

### C++

```bash
# From the repository root: configure and build the C++ library/tests.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure

# For a standalone recipe that links the already-built static library:
c++ -std=c++17 -Wall -Wextra -Wpedantic \
    -I cpp/include \
    recipe.cpp \
    cpp/build/libdpe.a \
    -o recipe
```

For runtime checking, benchmarks and profiling builds, use the reproducible
commands in [`cpp/README.md`](cpp/README.md). Build directories are intentionally
repository-relative; never copy a machine-specific path such as
`/Users/<name>/.../cpp/build` into source or documentation.

### Rust

Use a scratch crate with:

```toml
[dependencies]
dpe = { path = "../rust" }
```

Then:

```bash
cargo build --release
```

### Java

```bash
bash java/build.sh

javac -cp java/out/main RecipeN.java
cd java
java -cp out/main:.. RecipeN
```

---

## How these recipes are verified

The examples in this cookbook are executable tests, not pseudocode.

As part of the release check:

- all **12 Python snippets** are run against the Python reference implementation
- all **12 C++ snippets** are compiled with `-Wall -Wextra` and executed
- all **12 Rust snippets** are compiled and run against the Rust crate
- all **12 Java snippets** are compiled with `-Xlint:all -Werror` and executed

The deterministic examples produce the corresponding library results directly.

Monte Carlo output is not expected to be bit-identical across languages because each implementation uses its own RNG stream. Instead, results are compared statistically using their reported standard errors.

For the full behavioral contract, including validation rules and RNG guarantees, see `API_SPEC.md`.