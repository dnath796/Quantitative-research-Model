# WTI quantitative research framework

A reusable Python library for one-month-ahead return and second-moment forecasting,
built from the WTI work in `Crude_oil_regression.py` and `master_datasheet.xlsx`.
Researchers can change predictors, transformations, release-lag assumptions, model families,
and rolling/expanding windows through TOML configuration. The original coursework analyses
and the other projects in this repository remain available.

This is research software with automated checks. It is not a deployed trading system,
and historical forecast accuracy is not evidence of profitable execution.

## Install and run

Use Python 3.11–3.14. The lock file pins the runtime, build, and development environment.

```bash
python3 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .

# Offline deterministic demo; produces forecasts, metrics, diagnostics, and a report.
quant-research --config configs/demo.toml --output results/demo

# Existing WTI workbook, with identical specifications in both window modes.
quant-research --config configs/wti.toml --output results/wti-expanding
quant-research --config configs/wti.toml --window rolling --output results/wti-rolling
```

Choose a new output directory for each run. Existing results are never deliberately
overwritten; failed experiments do not publish a completed results directory.
The CLI exits with status 2 for expected configuration, file, and model-fit errors.
`python -m quant_research` exposes the same interface. With no `--config`, it runs a
seeded synthetic mean/OLS/GARCH experiment.

## Architecture and data flow

```text
configs/*.toml
    ↓
data.load_data         CSV / workbook / seeded synthetic panel
    ↓                  validate unique, ordered, contiguous calendar months
features.build_features
    ↓                  causal transforms → calendar lags → leading warmup removal
backtest.backtest
    ↓                  training-only standardization, refit each model every month
models.fit_model → forecast
    ↓                  mean + conditional variance + fit diagnostics
evaluation.evaluate    common test dates, mean losses, second-moment losses, DM comparisons
    ↓
pipeline.run_research  snapshot + metadata + report + optional cost-aware sign strategy
```

Code lives under `src/quant_research/{data,features,models,backtest,risk,utils}`.
`tests/unit` covers numerical behavior and validation; `tests/integration` covers the
research lifecycle, temporal leakage, CLI, artifact replay, and workbook adapter.
`notebooks/exploratory_analysis.ipynb` consumes the library rather than duplicating it.

## Reusable API

```python
from quant_research import (
    load_config, load_data, build_features, fit_model, forecast, backtest, evaluate,
)

config = load_config("configs/wti.toml")
panel = load_data(config.data, [f.column for f in config.features])
dataset = build_features(panel, config.features, config.data.target)

fitted = fit_model("ols", dataset.X.iloc[:120], dataset.y.iloc[:120])
prediction = forecast(fitted, dataset.X.iloc[120])
print(prediction.mean, prediction.variance)

result = backtest(dataset, config.models, config.backtest)
metrics, comparisons = evaluate(result.forecasts, config.backtest.dm_lags)
```

For a full saved experiment, use `run_research(config, "results/new-specification")`.
There is no live-data service or HTTP API; the reusable API is Python.

## Configuration and input contract

Paths in TOML are resolved relative to the config file. Unknown fields are errors.
For CSV input, provide a `date` column and the configured target/predictor columns.
For the existing Excel workbook, the adapter reads zero-based `header_row = 2`,
using its first unnamed column as the master date column. Units are never inferred
or rescaled by the loader. Only the target and requested predictors are selected.

```toml
models = ["mean", "ols", "garch", "gjr_garch"]

[data]
source = "csv"
path = "../data/monthly.csv"
target = "wti_chg_pct"
start = "2000-01"
end = "2025-12"

[backtest]
window = "rolling"
initial_train = 120
window_size = 120
dm_lags = 3
strategy = false
cost_bps = 5.0

[[features]]
column = "wti_chg_pct"
lag = 1

[[features]]
column = "indpro_chg_pct"
lag = 2
```

Feature transforms are `identity`, `difference`, or `log_return` (positive levels only).
Every feature has lag ≥ 1. A feature at target month t with lag k uses the transformed
observation at t−k. Dates are month labels, not observation-release timestamps.
The data must contain contiguous months; internal missing values fail instead of being
forward-filled or dropped. Only leading incomplete rows are removed after transformation.
Sample bounds apply before transformations, so include earlier history when needed.

In the supplied WTI specification, macro predictors have two-month lags as a conservative
availability assumption. These assumptions do not establish true release dates. The
workbook contains revised series, so this is **pseudo-out-of-sample** forecasting.
Real-time research requires vintage snapshots and audited release timestamps.

The workbook's `wti_chg_pct` values are decimal returns despite their names. For example,
the January 1990 value matches `(wti_close / wti_open) - 1`; this is not assumed to be a
continuous futures total-return series. The original script's same-month explanatory
regression and this package's next-month predictive regression answer different questions.
The default forecast specification deliberately uses a smaller, explicit predictor set.
It does not automatically repeat the original full-sample variable selection or causal claims.

## Models and numerical assumptions

| Model | Conditional mean | Conditional variance |
| --- | --- | --- |
| `mean` | Training-window average | Sample variance, ddof=1 |
| `ols` | Intercept + lagged features | Residual sum of squares / residual degrees of freedom |
| `garch` | Estimated constant | GARCH(1,1) |
| `gjr_garch` | Estimated constant | GJR-GARCH(1,1,1) |
| `egarch` | Estimated constant | EGARCH(1,1,1) |

All models are refitted at every origin. OLS feature centering and scaling are estimated
only on that origin's training window. The implementation rejects constant columns,
rank deficiency, insufficient residual degrees of freedom, and optimizer nonconvergence.
Mean/OLS variance uses a documented `1e-12` numerical floor with a diagnostics flag.
ARCH-family fits need at least 60 nonconstant observations and use Gaussian innovations.
Decimal returns are multiplied by 100 during fitting, with forecasts converted back by
100 for the mean and 10,000 for variance. No global warning suppression is installed.
Forecasts are analytic and restricted to **one month ahead**; no multi-horizon claim is made.
The implementation follows the [ARCH forecasting interface](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).

Rolling windows use exactly `window_size` training observations; expanding windows start
at the first usable observation. `initial_train` determines the first forecast target
for either mode. No model sees its forecast target or any later return while fitting.
All models are evaluated on exactly the same target dates. A failed fit stops the run,
rather than substituting a forecast or scoring different dates for different models.

## Evaluation and diagnostics

- **Mean forecasts:** RMSE, MAE, and signed forecast bias, in decimal-return units.
- **Second moments:** predict `m = variance + mean²` against the common proxy `return²`.
  QLIKE is `log(m) + return²/m`, omitting the model-independent term. This handles zero
  realized returns, can be negative, and is lower-is-better. Also report second-moment RMSE.
  This does not claim to observe or directly validate latent conditional variance.
- **Diebold–Mariano:** compare squared error, absolute error, and QLIKE against the
  historical-mean benchmark, with a Bartlett Newey–West long-run variance estimate.
  `dm_lags` is a prespecified HAC bandwidth; the statistic is mean loss difference divided
  by its estimated standard error. The reference distribution is asymptotic normal,
  two-sided, without small-sample correction. Negative favors the named model. Identical
  losses return p=1; insufficient samples or zero estimated variance return an explicit status.
  These are exploratory tests, with no multiple-testing or nested-model correction.
- **Per-fit diagnostics:** residual RMSE/autocorrelation, OLS condition number, coefficients,
  training scales, GARCH-family optimizer status, AIC/BIC, and parameter estimates.

The optional sign strategy takes `sign(predicted mean)` during each target interval.
It charges `cost_bps / 10000 × absolute position change`, including entry and final
liquidation; a direct long-to-short reversal costs two units. Cash earns zero. It stores
gross/net returns, turnover, wealth and drawdown, and stops if wealth is exhausted.
It is disabled for the WTI workbook by default because a tradable futures study needs
contract selection, roll, margin, financing, and execution rules.

## Stored results and reproducibility

Each new result directory contains:

| File | Purpose |
| --- | --- |
| `input.csv`, `config.json` | Exact selected panel and resolved specification |
| `forecasts.csv` | Target/origin dates, training bounds, sample sizes, actuals, mean, variance, second moment |
| `metrics.csv`, `comparisons.csv` | Model scores and DM comparisons |
| `diagnostics.jsonl` | Fit diagnostics for every model/origin |
| `metadata.json` | Source/input/code SHA-256, Git revision/dirty flag if available, dependency/Python/platform versions, runtime |
| `report.md`, `forecasts.png` | Offline report and forecast charts |
| `strategy.csv` | Optional illustrative strategy accounting |

Synthetic input is seeded. Models use deterministic analytic forecasts. Metadata timestamps
and timings naturally vary; numerical equality across different BLAS/OS versions is not
promised. For replay, point a CSV data config at the stored `input.csv` with the same
features/models/backtest settings. Results should match within numerical precision.

## Tests, packaging and CI

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest
python -m build --no-isolation
```

CI installs the package on Linux with Python 3.11, 3.12 and 3.14, runs lint/format checks
and the pytest suite, builds distributions, smoke-tests the installed wheel from outside
the repository, and uploads demo artifacts. A separate job builds and runs the Docker image.
Tests include future-data perturbations for both window modes, an independent OLS numerical
reference, the GARCH one-step recurrence and unit conversion, forecast-date alignment,
turnover costs, zero-return/constant-data cases, and explicit failure behavior.

GitHub Actions will run after these files are pushed. Local execution does not establish
that a remote CI run has passed.

## Linux / Docker

```bash
docker build -t wti-research .
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/results" wti-research --output /results/docker-demo

# Mount the repository read-only for workbook/config access.
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/research:ro" -v "$PWD/results:/results" \
  wti-research --config /research/configs/wti.toml --output /results/docker-wti
```

The runtime image runs as a nonroot user and uses a noninteractive plotting backend.
No credentials, private workbook, results, notebooks, or unrelated projects are embedded
in the image. Building requires access to the Python image registry and PyPI; running the
synthetic demo needs no network. AWS deployment and scheduling are not configured.

## Profiling and performance

```bash
python scripts/benchmark.py --config configs/demo.toml --output results/benchmark
python scripts/benchmark.py --config configs/wti.toml --output results/benchmark-wti
```

The script runs the real pipeline under `cProfile`, saves `profile.pstats`,
`profile.txt` (top cumulative costs), and `benchmark.json` with elapsed time and throughput.
These measurements include report generation and are machine-specific, not service latency
guarantees. See `docs/validation.md` for the execution evidence collected with this change.
Repeated GARCH optimization at every origin is expected to dominate larger experiments.
