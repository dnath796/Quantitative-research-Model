# Validation evidence

Executed locally on September 28, 2026 (America/New_York), using Python 3.14.4 on
macOS ARM64. Dependency versions are recorded in `requirements.lock`.

## Automated checks

- 63 pytest tests passed, including numerical references, future-data perturbations
  for rolling/expanding evaluation, validation failures, optimizer failure handling,
  turnover accounting, artifact replay, and the installed command-line interface.
- Ruff lint and format checks passed for `src`, `tests`, and `scripts`.
- `pip check` found no broken requirements in the isolated environment.
- Wheel and source distributions built successfully with `python -m build --no-isolation`.
- The built wheel was installed in a separate target directory and ran the demo from
  outside the repository, with its imported module path explicitly checked.
- All notebook code cells executed successfully with the noninteractive Agg backend.
- The WTI report plot was visually inspected for readable labels, legends, and dates.

The GitHub Actions workflow is configured for Linux Python 3.11, 3.12 and 3.14.
A remote CI run has not been triggered from this task.

## Linux container execution

The final Dockerfile built successfully. The resulting `wti-research` image ran the full
four-model synthetic specification as a nonroot user with `--network none`, producing
232 forecasts over 58 target months and writing a complete report to
`results/docker-demo`. The container's recorded source fingerprint matched the final
workspace source exactly. This verifies local Linux container execution; no cloud
deployment or scheduled service is claimed.

## Existing-workbook evaluation

Input: `master_datasheet.xlsx`, `Combined Monthly`, January 1991–December 2025.
420 raw monthly observations, 2 warmup observations, 120 initial training observations,
and 298 forecast target months from March 2001–December 2025. Each window mode produced
1,192 forecasts across historical mean, OLS, GARCH and GJR-GARCH. All fits converged.
Rolling windows contained 120 observations; expanding windows grew each month.

| Window | Model | Return RMSE | Return MAE | Second-moment QLIKE |
| --- | --- | ---: | ---: | ---: |
| Expanding | Historical mean | 0.107209 | 0.075861 | -3.390315 |
| Expanding | OLS | 0.109813 | 0.078052 | -3.400439 |
| Expanding | GARCH | 0.107099 | 0.075857 | -3.678436 |
| Expanding | GJR-GARCH | 0.107183 | 0.075926 | -3.698515 |
| Rolling | Historical mean | 0.107677 | 0.076069 | -3.372495 |
| Rolling | OLS | 0.123241 | 0.084194 | -3.513759 |
| Rolling | GARCH | 0.107326 | 0.075908 | -3.630311 |
| Rolling | GJR-GARCH | 0.107629 | 0.076146 | -3.657278 |

Returns are decimals, so 0.107209 RMSE corresponds to about 10.72 percentage points.
OLS did not improve return RMSE over the mean benchmark. In the expanding specification,
none of the nine exploratory DM comparisons was significant at an unadjusted 5% level.
These observations are specification-specific and do not establish tradable performance.
Macro-vintage and return-convention limitations are documented in the README and reports.

Local detailed artifacts:

- `results/wti-expanding/report.md`
- `results/wti-rolling/report.md`
- Each run's `forecasts.csv`, `comparisons.csv`, `diagnostics.jsonl`, and `metadata.json`

Workbook SHA-256:
`0b8223427424d9a6ac8be1bfcceaede3077ef31b416e46504326cdf8b932d6eb`.

## Performance

The profiled synthetic run used `configs/demo.toml`: 180 input observations, 58 forecast
months, and 4 models. It produced 232 model forecasts in **3.291 seconds**, approximately
70.5 model forecasts per second, including report generation and profiling overhead.
This is one measurement with a warm font cache, not a cold-start or service benchmark.
Model fitting took approximately 2.44 seconds cumulatively; numerical GARCH optimization
dominated the profile. The unprofiled workbook runs took approximately 10.71 seconds
(expanding) and 10.93 seconds (rolling), excluding Python import/startup time.

Profiler artifacts: `results/benchmark/{benchmark.json,profile.txt,profile.pstats}`.
Timing varies with CPU, BLAS, operating system, cache state and concurrent activity.
