import numpy as np
import pandas as pd
import pytest

from quant_research.errors import ModelFitError, ResearchError
from quant_research.models import fit_model


def test_historical_mean_numerical_reference(dataset):
    fitted = fit_model("mean", dataset.X, dataset.y)
    prediction = fitted.forecast()
    assert prediction.mean == pytest.approx(dataset.y.mean())
    assert prediction.variance == pytest.approx(dataset.y.var(ddof=1))


def test_ols_matches_known_linear_system(dataset):
    X = dataset.X
    y = 0.03 + 0.7 * X.iloc[:, 0] - 0.4 * X.iloc[:, 1]
    fitted = fit_model("ols", X.iloc[:-1], y.iloc[:-1])
    assert fitted.forecast(X.iloc[-1]).mean == pytest.approx(y.iloc[-1], abs=1e-12)
    assert fitted.variance > 0
    with pytest.raises(ResearchError, match="column order"):
        fitted.forecast(X.iloc[-1].iloc[::-1])


def test_ols_matches_independent_unscaled_normal_equations(dataset):
    fitted = fit_model("ols", dataset.X.iloc[:-1], dataset.y.iloc[:-1])
    train = np.column_stack([np.ones(len(dataset.y) - 1), dataset.X.iloc[:-1]])
    coefficients = np.linalg.solve(train.T @ train, train.T @ dataset.y.iloc[:-1].to_numpy())
    row = np.r_[1, dataset.X.iloc[-1].to_numpy()]
    assert fitted.forecast(dataset.X.iloc[-1]).mean == pytest.approx(row @ coefficients)
    residuals = dataset.y.iloc[:-1].to_numpy() - train @ coefficients
    assert fitted.variance == pytest.approx(residuals @ residuals / (len(train) - 3))


def test_rank_deficiency_and_constant_feature(dataset):
    X = dataset.X.copy()
    X["duplicate"] = X.iloc[:, 0]
    with pytest.raises(ModelFitError, match="rank deficient"):
        fit_model("ols", X, dataset.y)
    X["constant"] = 1
    with pytest.raises(ModelFitError, match="constant feature"):
        fit_model("ols", X, dataset.y)


@pytest.mark.parametrize("name", ["garch", "gjr_garch", "egarch"])
def test_arch_family_positive_one_step_forecasts(dataset, name):
    model = fit_model(name, dataset.X, dataset.y)
    prediction = model.forecast()
    assert np.isfinite(prediction.mean)
    assert prediction.variance > 0
    assert model.diagnostics["convergence_flag"] == 0


def test_garch_recursion_and_scaling(dataset):
    from arch import arch_model

    reference = arch_model(
        dataset.y.to_numpy() * 100, mean="Constant", vol="GARCH", p=1, q=1, rescale=False
    ).fit(disp="off", options={"maxiter": 1000})
    p = reference.params
    # ARCH forecasting recomputes the backcast from fitted residuals; the fit's stored
    # conditional_volatility instead uses the initial estimation backcast.
    backcast = reference.model.volatility.backcast(reference.resid)
    variance = p["omega"] + (p["alpha[1]"] + p["beta[1]"]) * backcast
    for residual in reference.resid:
        variance = p["omega"] + p["alpha[1]"] * residual**2 + p["beta[1]"] * variance
    expected = variance / 10000
    actual = fit_model("garch", dataset.X, dataset.y).forecast()
    assert actual.variance == pytest.approx(expected, rel=1e-8)
    assert actual.mean == pytest.approx(p["mu"] / 100)


def test_degenerate_returns_and_unaligned_inputs(dataset):
    y = pd.Series(0.0, index=dataset.y.index)
    assert fit_model("mean", dataset.X, y).forecast().variance > 0
    with pytest.raises(ModelFitError, match="nonconstant"):
        fit_model("garch", dataset.X, y)
    with pytest.raises(ResearchError, match="indices"):
        fit_model("mean", dataset.X.iloc[1:], dataset.y)


def test_optimizer_failure_is_not_silently_scored(monkeypatch, dataset):
    from types import SimpleNamespace

    import quant_research.models as models

    result = SimpleNamespace(
        convergence_flag=9, optimization_result=SimpleNamespace(message="did not converge")
    )
    monkeypatch.setattr(
        models, "arch_model", lambda *a, **k: SimpleNamespace(fit=lambda **kw: result)
    )
    with pytest.raises(ModelFitError, match="optimizer failed"):
        fit_model("garch", dataset.X, dataset.y)
