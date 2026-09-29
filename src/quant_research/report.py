"""Standalone research report and scientific forecast plots."""

from pathlib import Path

import pandas as pd


def write_report(
    directory: Path, forecasts: pd.DataFrame, metrics: pd.DataFrame, comparisons: pd.DataFrame
) -> None:
    """Render an offline Markdown report with a PNG generated via Matplotlib's Agg backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True)
    actual = forecasts[forecasts.model == "mean"]
    top.plot(actual.target_date, actual.actual, color="#64748b", alpha=0.7, label="Observed return")
    for name, group in forecasts.groupby("model"):
        top.plot(group.target_date, group["mean"], label=name, linewidth=1)
        bottom.plot(group.target_date, group.second_moment, label=name, linewidth=1)
    top.set(ylabel="Monthly return (decimal)", title="One-month-ahead out-of-sample forecasts")
    bottom.set(ylabel="Predicted E[return²]", xlabel="Target month")
    for axis in (top, bottom):
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
    figure.savefig(directory / "forecasts.png", dpi=140)

    def table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            cells = [f"{x:.6g}" if isinstance(x, float) else str(x) for x in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    text = f"""# WTI forecast research run

Target months: {actual.target_date.min():%Y-%m} to {actual.target_date.max():%Y-%m}.
Each model was refitted at every origin using only earlier target observations.
Returns retain the source's decimal units. See `config.json` and `metadata.json` for provenance.

## Forecast accuracy

{table(metrics)}

Lower RMSE, MAE and QLIKE are better. QLIKE is log(m) + r²/m with m = variance + mean².
It evaluates the same squared-return proxy for every model, not observed latent variance.
The omitted proxy-only term means QLIKE may be negative. Bias is prediction minus actual.

## Comparisons with historical mean

{table(comparisons)}

Negative DM statistics favor the named model. P-values use an asymptotic two-sided normal
reference and Bartlett HAC with the configured lag count. They are exploratory, unadjusted
for multiple comparisons, and unreliable for small samples or nested model selection.
No significance claim establishes a tradable advantage.

![Forecasts](forecasts.png)

## Assumptions and limitations

Predictors are transformed causally and shifted by their configured calendar-month lags.
Those lags are availability assumptions, not verified release timestamps. Revised macro data
can retain vintage leakage. This is pseudo-out-of-sample evaluation, not a real-time vintage study.
The existing workbook's return conventions and provenance require independent audit before trading.
GARCH-family estimates use a constant mean and Gaussian innovations. Squared monthly returns
are a noisy second-moment proxy. No hyperparameter tuning is performed on the test sample.
Any `strategy.csv` is an illustrative sign strategy with entry, reversal and final-exit costs;
it omits futures roll, funding, margin and execution constraints.

`diagnostics.jsonl` records per-fit residual diagnostics, OLS scaling/coefficients, and ARCH
optimizer status/parameters. `input.csv` stores the actual input panel used in this run.
"""
    (directory / "report.md").write_text(text, encoding="utf-8")
