# DerivativeLab Web App

Interactive derivatives pricing and risk dashboard for the coursework project.

Live site: [dnath796.github.io/Quantitative-research-Model](https://dnath796.github.io/Quantitative-research-Model/)

## Features

- Black–Scholes, Cox–Ross–Rubinstein binomial, and Monte Carlo pricing
- Calls, puts, option strategies, payoff/P&L views, and futures analytics
- Options-chain table with bid, ask, last, implied volatility, volume, open interest, and Greeks
- Delta, gamma, vega, theta, rho, scenario analysis, and portfolio-level risk
- Implied-volatility analysis, convergence validation, and delta-hedging simulation

## Local development

Node.js 22 or newer is recommended.

```bash
npm ci
npm run dev
```

## GitHub Pages build

```bash
npm run build:pages
npm run preview:pages
```

The static site is written to `dist-pages/`. The repository workflow at
`.github/workflows/derivative-lab-pages.yml` builds and deploys that directory
whenever the `main` branch changes.

The project-name base path is configured in `vite.pages.config.ts` as
`/Quantitative-research-Model/`, matching the GitHub repository name.
