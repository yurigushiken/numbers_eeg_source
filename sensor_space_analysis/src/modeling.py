"""Mixed-effects modeling for RT ~ left temporal amplitude × accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults


def build_mixed_model(df: pd.DataFrame) -> MixedLM:
    """Build a mixed model on correct trials only:

    RT ~ left_temp_amp_cws + (1 + left_temp_amp_cws | subject_id)

    Expects `left_temp_amp_cws` already centered within subject.
    """
    data = df.copy()
    data['accuracy'] = data['accuracy'].astype(int)
    data = data[data['accuracy'] == 1].copy()
    md = smf.mixedlm(
        formula="rt ~ left_temp_amp_cws",
        data=data,
        groups=data["subject_id"],
        re_formula="~ left_temp_amp_cws",
    )
    return md


def fit_mixed_model(model: MixedLM) -> MixedLMResults:
    """Fit the mixed model with a robust optimizer and return the result."""
    res = model.fit(method="lbfgs", maxiter=200, disp=False)
    return res


def extract_fixed_effects(res: MixedLMResults) -> pd.DataFrame:
    """Return a tidy fixed-effects table with term, estimate, std err, t, p."""
    params = res.fe_params
    bse = res.bse_fe
    tvals = params / bse
    pvals = res.pvalues.loc[params.index]
    out = pd.DataFrame(
        {
            'term': params.index,
            'estimate': params.values,
            'std_err': bse.values,
            't': tvals.values,
            'p': pvals.values,
        }
    )
    return out


def extract_random_effects(res: MixedLMResults) -> pd.DataFrame:
    """Return subject-specific random effects (intercept and slope)."""
    rows = []
    for sid, re in res.random_effects.items():
        # Random effects include intercept and slope(s)
        vals = {'subject_id': sid}
        for k, v in re.items():
            vals[k] = float(v)
        rows.append(vals)
    out = pd.DataFrame(rows)
    # Ensure slope column name exists
    if 'left_temp_amp_cws' not in out.columns:
        out['left_temp_amp_cws'] = 0.0
    if 'Intercept' not in out.columns:
        out['Intercept'] = 0.0
    return out[['subject_id', 'Intercept', 'left_temp_amp_cws']]


def _marginal_fixed_only_predictions(df: pd.DataFrame, res: MixedLMResults) -> np.ndarray:
    """Compute fixed-effects-only predictions for RT (marginal)."""
    cols = df.columns.tolist()
    if 'accuracy' in cols and 'accuracy' in res.model.data.param_names:
        fe_model = smf.ols("rt ~ left_temp_amp_cws * accuracy", data=df)
    else:
        fe_model = smf.ols("rt ~ left_temp_amp_cws", data=df)
    design = fe_model.exog
    beta = res.fe_params.values
    return design @ beta


def model_diagnostics(res: MixedLMResults) -> str:
    """Return a diagnostics string with R2 (approx), AIC, BIC, residual SD."""
    y = res.model.endog
    yhat = res.fittedvalues  # includes random effects
    resid = y - yhat
    aic = res.aic
    bic = res.bic
    # Conditional R2: with random effects
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_cond = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Marginal R2: fixed-effects only
    try:
        yhat_fix = _marginal_fixed_only_predictions(pd.DataFrame(res.model.data.frame), res)
        ss_res_fix = float(np.sum((y - yhat_fix) ** 2))
        r2_marg = 1.0 - ss_res_fix / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        r2_marg = np.nan
    s = float(np.sqrt(np.mean(resid ** 2)))
    return (
        f"AIC: {aic:.3f}\n"
        f"BIC: {bic:.3f}\n"
        f"R2 (marginal): {r2_marg:.3f}\n"
        f"R2 (conditional): {r2_cond:.3f}\n"
        f"Residual SD: {s:.3f}\n"
    )


def generate_predictions(df: pd.DataFrame, res: MixedLMResults) -> pd.DataFrame:
    """Generate fixed-effects predictions across a grid of left_temp_amp_cws.

    If the model includes accuracy, returns separate curves; otherwise returns a single curve.
    """
    grid = np.linspace(df['left_temp_amp_cws'].min(), df['left_temp_amp_cws'].max(), 25)
    if 'accuracy' in res.model.data.frame.columns and 'accuracy' in res.model.data.param_names:
        rows = []
        for acc in (0, 1):
            X = pd.DataFrame({'left_temp_amp_cws': grid, 'accuracy': acc, 'rt': 0.0})
            yhat = _marginal_fixed_only_predictions(X, res)
            rows.append(pd.DataFrame({'left_temp_amp_cws': grid, 'accuracy': acc, 'rt_pred': yhat}))
        return pd.concat(rows, ignore_index=True)
    else:
        X = pd.DataFrame({'left_temp_amp_cws': grid, 'rt': 0.0})
        yhat = _marginal_fixed_only_predictions(X, res)
        return pd.DataFrame({'left_temp_amp_cws': grid, 'rt_pred': yhat})
