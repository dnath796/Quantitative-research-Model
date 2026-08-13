# Step-by-Step Coursework Roadmap

This map takes the project from financial foundations to a complete derivatives pricing and risk report. Each stage produces something demonstrable, testable, and suitable for inclusion in a coursework submission.

## Stage 1 — Market vocabulary and Python foundations

**Learn:** underlying assets, spot and forward prices, strike, maturity, risk-free rate, dividend yield, volatility, compounding, present value, and future value.

**Build:** validated `MarketData` and enumerations for option type and position direction.

**Experiment:** convert annual percentage inputs to decimals, discount a future cash flow, and compare simple with continuous compounding.

**Evidence:** a definitions table, small calculations, and validation tests.

## Stage 2 — Derivative contracts and market structure

**Learn:** forwards, futures, calls, puts, swaps; exchange-traded versus over-the-counter markets; rights versus obligations; counterparty and margin risk.

**Build:** `Derivative`, `Option`, `Forward`, `Future`, and `FixedForFloatingSwap`.

**Discuss:** hedgers, speculators, arbitrageurs, market makers, banks, corporations, institutions, and portfolio managers. Explain hedging, speculation, income, price discovery, risk transfer, and arbitrage.

**Evidence:** a class diagram and two real-world use cases per major derivative type.

## Stage 3 — Option lifecycle and basic payoffs

**Learn:** contract specifications, premium, holder/writer, opening purchase/sale, closing sale/purchase, exercise, assignment, expiration, and moneyness.

**Build:** vectorized call and put payoffs plus long/short profit functions.

**Experiment:** calculate values at five terminal prices around the strike.

**Evidence:** explicitly show why payoff and P&L differ by the premium.

## Stage 4 — Tables and diagrams

**Learn:** maximum profit, maximum loss, break-even, slope, and nonlinear exposure.

**Build:** Pandas P&L tables and Matplotlib payoff/P&L plots.

**Experiment:** compare long call, short call, long put, and short put on identical axes.

**Evidence:** labeled plots with the strike, zero-profit line, break-even, and a written interpretation.

## Stage 5 — Strategies, spreads, and income

**Learn:** how position legs combine and how a trade expresses a market view.

**Build in order:**

1. Covered call and protective put.
2. Bull call and bear put debit spreads.
3. Bull put and bear call credit spreads.
4. Straddle and strangle.
5. Collar and butterfly.
6. Iron condor.

**Experiment:** alter one strike or premium and explain the changed break-even and risk limits.

**Evidence:** for every selected strategy, include its objective, legs, P&L table, plot, maximum gain/loss, and break-even points. Identify the covered call as combined call writing and explain its income/upside trade-off.

## Stage 6 — Futures and cost of carry

**Learn:** contract standardization, clearing, margin, daily settlement, convergence, basis, long/short payoff, and cash-and-carry pricing.

**Build:** futures payoff and `F_0 = S_0 exp((r-q)T)`.

**Experiment:** shock the financing rate, dividend yield, and maturity.

**Evidence:** one stock-index hedge example and a discussion of basis risk.

## Stage 7 — No-arbitrage and replication

**Learn:** the law of one price, short selling, replication, synthetic positions, and risk-neutral valuation.

**Build:** current forward value, put-call parity, parity-implied prices, and a signed parity gap.

**Experiment:** introduce a deliberately mispriced call and describe the offsetting trade, then explain why transaction costs could remove the opportunity.

**Evidence:** a cash-flow table at expiration demonstrating identical replicating payoffs.

## Stage 8 — One-period and multi-period binomial trees

**Learn:** up/down factors, risk-neutral probability, expected risk-neutral payoff, discounting, and backward induction.

**Build in order:**

1. Draw and calculate a one-step tree by hand.
2. Inspect the engine’s two-step stock and option trees.
3. Price European calls and puts.
4. Add early-exercise comparison for American contracts.
5. Increase steps through 5, 25, 50, 100, and 500.

**Evidence:** node calculations, a no-arbitrage probability check, and an example showing the American put value is at least the European put value.

## Stage 9 — Black–Scholes–Merton

**Learn:** `d1`, `d2`, risk-neutral probability interpretation, continuous discounting, model inputs, and assumptions.

**Build:** European call and put functions with dividend yield and boundary cases.

**Experiment:** change spot, strike, maturity, rate, yield, and volatility separately.

**Evidence:** verify put-call parity and compare at least one result to a textbook value. Critique constant volatility, continuous trading, frictionless markets, and lognormal returns.

## Stage 10 — Greeks and sensitivities

**Learn:** delta, gamma, theta, vega, rho, first-order approximation, convexity, and units.

**Build:** analytical Greeks and curves against spot; add variable sensitivity tables.

**Experiment:** validate delta, gamma, and vega using finite differences. Contrast an at-the-money option with deep in- and out-of-the-money options.

**Evidence:** five Greek plots and an explanation of what each means for hedging and P&L.

## Stage 11 — Portfolio risk and delta hedging

**Learn:** signed quantities, option contract multipliers, aggregation, delta neutrality, hedge rebalancing, and residual risks.

**Build:** a portfolio containing calls, puts, and stock; aggregate all five Greeks; calculate the stock hedge.

**Experiment:** hedge at the initial spot, move the spot, and show that gamma causes delta to change.

**Evidence:** before/after hedge tables and a discussion of gamma, vega, theta, liquidity, transaction-cost, and model risk.

## Stage 12 — Implied volatility, scenarios, and model comparison

**Learn:** solving price-to-volatility inversions, no-arbitrage price bounds, full revaluation, and numerical convergence.

**Build:** bracketed implied volatility, stock/volatility/rate/time scenarios, and binomial-versus-BSM convergence charts.

**Experiment:** recover a known volatility from a model price; shock stock ±10%, volatility ±5 points, rates +1 point, and advance time 30 days.

**Evidence:** a scenario table with value and P&L plus a convergence plot and interpretation.

## Stage 13 — Monte Carlo, volatility surfaces, and dynamic hedging

**Learn:** simulation error, confidence intervals, variance reduction, volatility smiles/skews, surface calibration, discrete rebalancing, realized versus implied volatility, and transaction-cost drag.

**Build:** antithetic Monte Carlo pricing, analytical-versus-simulation validation, quote-based implied-volatility surfaces, a dynamic delta-hedging simulator, and a hedging P&L distribution.

**Experiment:** increase simulation paths and hedge frequency separately; change realized volatility relative to implied volatility; add transaction costs; explain the effects on estimator error and hedge P&L.

**Evidence:** confidence interval and convergence tables, a strike/maturity surface, hedge P&L percentiles, and one complete rebalance history.

## Stage 14 — Integrated final submission

**Run:**

```bash
python main.py --spot 100 --strike 105 --maturity 0.5 \
  --rate 0.05 --volatility 0.25 --dividend-yield 0.02 \
  --option-type call --position long --quantity 10 \
  --output output/final
```

**Submit:**

- source code with docstrings and tests;
- a report explaining theory, assumptions, methods, results, and limitations;
- payoff/P&L, Greek, and convergence figures;
- scenario CSV and interpretation;
- reproducibility instructions;
- a short demonstration using the CLI;
- an appendix showing that the automated test suite passes.

## Suggested 12-week schedule

| Week | Work |
|---:|---|
| 1 | Stages 1–2: terminology, market structure, data and contract classes |
| 2 | Stage 3: option lifecycle, payoff, long and short profit |
| 3 | Stage 4: tables, diagrams, break-even, maximum gain/loss |
| 4 | Stage 5: covered/protective strategies and vertical spreads |
| 5 | Stage 5: volatility/range strategies; compare trade-offs |
| 6 | Stages 6–7: futures, carry, no-arbitrage, replication, parity |
| 7 | Stage 8: one- and multi-period European binomial trees |
| 8 | Stage 8: American exercise and convergence |
| 9 | Stage 9: Black–Scholes implementation, assumptions, sensitivities |
| 10 | Stage 10: analytical and finite-difference Greeks |
| 11 | Stages 11–12: portfolio risk, hedging, IV, scenarios, comparison |
| 12 | Stages 13–14: Monte Carlo, surfaces, hedging, integrated run and presentation |

## Completion checklist

- [ ] I can define each major derivative and market participant.
- [ ] I can distinguish option payoff, premium cash flow, and profit.
- [ ] I can explain opening, closing, exercise, assignment, and expiration.
- [ ] I can construct and interpret common option strategies.
- [ ] I can derive a one-step binomial price and explain risk-neutral probability.
- [ ] I can state the Black–Scholes assumptions and interpret every input.
- [ ] I can calculate, validate, and interpret all five principal Greeks.
- [ ] I can aggregate position risk and construct a delta hedge.
- [ ] I can price a basic future and explain cost of carry.
- [ ] I can perform implied-volatility, scenario, and convergence analysis.
- [ ] I can run the complete engine and reproduce its report.
- [ ] I can explain Monte Carlo confidence intervals and convergence.
- [ ] I can construct or interpret an implied-volatility surface.
- [ ] I can simulate delta rebalancing and explain hedging P&L.
