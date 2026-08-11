# Derivatives Analytics Report

## Contract

| Field | Value |
|---|---:|
| Contract | European Call |
| Position | Long 10 contract(s) |
| Contract multiplier | 100 |
| Spot | $200.00 |
| Strike | $105.00 |
| Maturity | 0.5000 years |
| Volatility | 25.00% |
| Risk-free rate | 5.00% |
| Dividend yield | 2.00% |

## Pricing

| Model | Value per unit |
|---|---:|
| Black-Scholes | $95.6030 |
| CRR binomial (100 steps) | $95.6029 |
| CRR binomial (500 steps) | $95.6030 |

## Greeks

Vega and rho are shown per 1.00 change in volatility/rate; divide by 100 for a one-point move.

| Greek | Per unit | Position |
|---|---:|---:|
| Delta | 0.989983 | 989.9833 |
| Gamma | 0.000008 | 0.0076 |
| Theta (annual) | -1.169282 | -1169.2819 |
| Vega | 0.038128 | 38.1280 |
| Rho | 51.196831 | 51196.8308 |

## Expiration Risk

| Metric | Value |
|---|---:|
| Premium used | $95.6030 per unit |
| Maximum profit | Unlimited |
| Maximum loss | $95,603.00 |
| Break-even | $200.60 |

## Charts and Scenarios

![Payoff and P&L](payoff_pnl.png)

![Greeks](greeks.png)

![Model convergence](convergence.png)

Detailed scenario results are in `scenarios.csv`.
