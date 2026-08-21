# Derivatives Pricing, Strategy and Risk Analytics Engine

A coursework-ready Python project that develops derivatives knowledge in the same order that the financial ideas depend on one another: contracts first, then payoffs, positions, strategies, no-arbitrage, numerical pricing, closed-form pricing, Greeks, portfolios, and finally integrated risk analysis.

Interactive dashboard: [dnath796.github.io/Quantitative-research-Model](https://dnath796.github.io/Quantitative-research-Model/)

## Learning-outcome map

| Stage | Financial topic | What to learn | Project evidence |
|---:|---|---|---|
| 1 | Foundations | Spot, strike, maturity, rates, yield, volatility, discounting | `MarketData` and validated domain types |
| 2 | Derivative markets | Forwards, futures, options, swaps; participants and uses | Contract classes and the notes below |
| 3 | Option basics | Calls, puts, long/short positions, opening and closing | `Option`, `OptionPosition`, payoff functions |
| 4 | Payoff vs P&L | Premium cash flow, break-even, maximum gain/loss | `payoffs.py`, strategy tables and summaries |
| 5 | Strategies | Hedging, speculation, income, combinations and spreads | Covered call through iron condor factories |
| 6 | Futures | Long/short payoff and cost-of-carry pricing | `Future`, `futures_payoff`, `futures_price` |
| 7 | No-arbitrage | Replication, risk-neutral valuation, put-call parity | Forward value and parity gap functions |
| 8 | Binomial model | CRR up/down factors, risk-neutral probability, backward induction | European and American lattice pricer |
| 9 | Black–Scholes | Inputs, assumptions, `d1`, `d2`, calls and puts | Robust BSM implementation |
| 10 | Greeks | Delta, gamma, theta, vega and rho | Analytical per-security Greeks and charts |
| 11 | Portfolio risk | Quantity/multiplier aggregation and delta hedging | `Portfolio` and `delta_hedge` |
| 12 | Applied analytics | Scenarios, sensitivities, implied volatility, volatility surfaces, Monte Carlo, model comparison | Scenario tables, surfaces, simulation, IV and convergence |
| 13 | Trading simulation | Dynamic delta rebalancing, transaction costs, realized volatility, hedge P&L | Hedging paths and P&L distribution |
| 14 | Final system | Reproducible dashboard-style report and live webpage | CLI report plus published interactive workbench |

This order is deliberate: a price is meaningful only after the contract and its cash flows are understood, and a Greek is meaningful only after a pricing model exists.

## Conceptual foundation

### Major derivative types

- **Forward:** private agreement to buy or sell an asset later at a price fixed today. It is customized and carries counterparty risk.
- **Future:** standardized, exchange-traded forward-like contract with margining and daily settlement.
- **Option:** gives its buyer a right, but not an obligation, to buy (call) or sell (put). The writer has the corresponding obligation.
- **Swap:** agreement to exchange cash-flow streams, such as fixed interest for floating interest.

Options trace their modern exchange-traded form to the growth of organized options markets in the 1970s, though option-like agreements are much older. A holder opens a long position by buying; a writer opens a short position by selling. Either can close before expiration with the opposite transaction in the same contract. Exercise uses the contractual right, assignment requires the writer to perform, and expiration ends an unexercised contract.

### Participants and uses

| Participant | Typical objective |
|---|---|
| Hedger | Reduce an existing price, rate, currency, or volatility exposure |
| Speculator | Seek leveraged profit from a market view while accepting risk |
| Arbitrageur | Exploit inconsistent prices with offsetting trades |
| Market maker | Quote two-sided markets and manage inventory risk |
| Bank/institution | Intermediate, structure products, hedge, or manage portfolios |
| Corporation | Stabilize input costs, revenues, funding rates, or exchange rates |
| Portfolio manager | Adjust exposure, protect downside, or generate income |

Examples include a protective put for downside insurance, a long call for bullish speculation, a covered call for premium income, an equity-index future for portfolio beta management, and put-call parity trades for arbitrage.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the integrated example:

```bash
python main.py \
  --spot 100 --strike 105 --maturity 0.5 \
  --rate 0.05 --volatility 0.25 --dividend-yield 0.02 \
  --option-type call --position long --quantity 10 \
  --output output
```

The command prints key values and creates:

```text
output/
├── report.md
├── scenarios.csv
├── payoff_pnl.png
├── greeks.png
└── convergence.png
```

Run the tests with `python -m pytest`.

## Guided coursework sequence

### 1. Contracts and basic cash flows

Start with `MarketData`, `Option`, `Forward`, `Future`, and `FixedForFloatingSwap` in `derivatives_engine/models.py`. For options, distinguish the terminal payoff from profit:

```text
call payoff = max(S_T - K, 0)
put payoff  = max(K - S_T, 0)
long option profit = payoff - premium
short option profit = premium - payoff
```

Explore these with `call_payoff`, `put_payoff`, `option_profit`, and `pnl_table`.

### 2. Strategies and combined call writing

`OptionStrategy` adds position profit arrays element by element and derives a P&L table, break-even points, maximum profit, and maximum loss. Included factories are:

- covered call and protective put;
- bull call, bear put, bull put, and bear call spreads;
- straddle, strangle, collar, butterfly, and iron condor.

A covered call is combined call writing: long the underlying and short a call. It earns premium and caps upside above the strike, while retaining substantial downside exposure.

### 3. Futures and no-arbitrage

With continuous compounding and dividend yield `q`:

```text
F_0 = S_0 exp((r-q)T)
```

European put-call parity is:

```text
C - P = S_0 exp(-qT) - K exp(-rT)
```

The project reports a signed parity gap. A material nonzero gap suggests inconsistent inputs; a real trading conclusion must still account for bid/ask spreads, fees, short-sale constraints, funding, dividends, and execution risk.

### 4. Binomial pricing

The Cox–Ross–Rubinstein tree uses:

```text
dt = T / n
u  = exp(sigma sqrt(dt))
d  = 1/u
p  = [exp((r-q)dt) - d] / (u-d)
```

Terminal intrinsic values are discounted backward under the risk-neutral probability. For American contracts, each node takes the larger of intrinsic value and continuation value. Use `return_tree=True` on a small tree to inspect every stock and option node, then increase the steps to study convergence.

### 5. Black–Scholes–Merton

The model assumes frictionless trading, no arbitrage, continuous trading and compounding, a constant risk-free rate and volatility, lognormal underlying prices, and known continuous dividend yield. The closed form is for European exercise; American early-exercise features require a lattice or another numerical method.

Inputs have clear effects but should not be memorized without context: higher spot generally raises calls and lowers puts; higher strike does the opposite; higher volatility increases both; rates usually raise calls and lower puts; dividends usually lower calls and raise puts. Time effects can be less simple for dividend-paying or deeply in-the-money options.

### 6. Greeks and portfolio risk

- **Delta:** first-order price sensitivity to spot.
- **Gamma:** rate of change of delta; curvature with respect to spot.
- **Theta:** annual calendar-time decay in this project.
- **Vega:** sensitivity to a 1.00 volatility change; divide by 100 for one percentage point.
- **Rho:** sensitivity to a 1.00 rate change; divide by 100 for one percentage point.

Portfolio Greeks are quantity-weighted sums. Contract multipliers matter: ten listed equity option contracts commonly represent 1,000 underlying units. `Portfolio.delta_hedge` returns the stock trade that offsets current delta; the hedge is local and changes as spot, volatility, and time change.

### 7. Complete analytics model

The final CLI links the full workflow:

```text
Validated inputs
  -> contract and position
  -> Black-Scholes and binomial values
  -> security and position Greeks
  -> expiration risk
  -> shocked revaluation scenarios
  -> payoff, Greek, and convergence charts
  -> reproducible report
```

For a written submission, discuss why binomial values oscillate around and converge toward Black–Scholes, when early exercise matters, which assumptions are least realistic, and why delta-only hedging leaves gamma, vega, theta, basis, liquidity, and model risk.

## Python API examples

```python
from derivatives_engine import MarketData, Option, black_scholes
from derivatives_engine.pricing import binomial_option_price, implied_volatility
from derivatives_engine.risk import Portfolio, OptionHolding, StockHolding

market = MarketData(spot=100, rate=0.05, volatility=0.20, dividend_yield=0.01)
call = Option(strike=100, maturity=1.0, option_type="call")

bs = black_scholes(100, 100, 1, 0.05, 0.20, "call", 0.01)
tree = binomial_option_price(100, 100, 1, 0.05, 0.20, 500, "call", 0.01)
iv = implied_volatility(bs, 100, 100, 1, 0.05, "call", 0.01)

portfolio = Portfolio().add_option(OptionHolding(call, quantity=10, contract_multiplier=100))
shares_to_trade = portfolio.delta_hedge(market)
```

See `examples/coursework_walkthrough.py` for an end-to-end console walkthrough.
The detailed weekly build and submission plan is in `COURSEWORK_ROADMAP.md`.

## Scope and extensions

This is an educational analytics engine, not a production trading system. Natural extensions are finite-difference methods, discrete dividends, multi-asset derivatives, value at risk, live option-chain calibration, and market-data integration.

## Monte Carlo, volatility surface, and trading simulation

The completed engine also includes:

- antithetic Monte Carlo pricing with standard error and confidence intervals;
- analytical-versus-Monte-Carlo validation and simulation convergence tables;
- market-quote implied-volatility surface inversion and matrix output;
- an illustrative skew/smile/term-structure surface generator for teaching;
- dynamic delta hedging of a short European option under GBM paths;
- configurable hedge frequency, realized volatility, dividends, and transaction costs;
- hedge P&L mean, dispersion, percentiles, and a complete sample rebalance path.
- a model-generated options chain containing calls and puts across expirations;
- bid, ask, midpoint, last, spread, volume, open interest, implied volatility,
  all five Greeks, intrinsic/extrinsic value, and market-model comparison;
- chain filtering by expiration, option type, and moneyness.

```python
from derivatives_engine import monte_carlo_european, simulate_delta_hedge

mc = monte_carlo_european(100, 105, 0.5, 0.05, 0.25, simulations=100_000)
hedge = simulate_delta_hedge(
    100, 105, 0.5, 0.05, 0.25,
    realized_volatility=0.28,
    hedge_steps=52,
    paths=2_000,
    transaction_cost_bps=1,
)
```

```python
from derivatives_engine import generate_options_chain

chain = generate_options_chain(
    spot=100,
    rate=0.05,
    base_volatility=0.25,
    dividend_yield=0.02,
)
```

The generated bid/ask and liquidity fields are reproducible simulated data for
coursework and interface development. They must not be interpreted as live
exchange quotes.
