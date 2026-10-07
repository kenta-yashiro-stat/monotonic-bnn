from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from numpyro.diagnostics import summary as diagnostic_summary
from numpyro.infer import MCMC

# Compute the 95% credible interval for a given set of samples.
def _interval(samples: np.ndarray, level: float = 0.95):
    alpha = (1.0 - level) / 2.0
    return np.quantile(samples, [alpha, 1.0 - alpha], axis=0)


def _crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n = samples.shape[0]
    sorted_samples = np.sort(samples, axis=0)
    coefficients = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    first_term = np.mean(np.abs(samples - truth[None, :]), axis=0)
    second_term = np.sum(coefficients * sorted_samples, axis=0) / n**2
    return first_term - second_term

# Evaluate the performance of the model in a specific region.
def evaluate_region(
    latent_draws: np.ndarray,
    sigma_draws: np.ndarray,
    derivative_draws: np.ndarray,
    f_true: np.ndarray,
    noise_sd_true: float,
    rng: np.random.Generator,
    predictive_mc_draws: int = 1,
) -> dict[str, float]:
    latent_mean = latent_draws.mean(axis=0)
    latent_low, latent_high = _interval(latent_draws)

    expanded_latent = np.repeat(latent_draws, predictive_mc_draws, axis=0)
    expanded_sigma = np.repeat(sigma_draws, predictive_mc_draws)
    predictive = expanded_latent + rng.normal(size=expanded_latent.shape) * expanded_sigma[:, None]
    pred_low, pred_high = _interval(predictive)

    y_eval = f_true + rng.normal(0.0, noise_sd_true, size=f_true.shape)
    residual = latent_mean - f_true
    log_density_draws = (
        -0.5 * np.log(2.0 * np.pi * sigma_draws[:, None] ** 2)
        -0.5 * ((y_eval[None, :] - latent_draws) / sigma_draws[:, None]) ** 2
    )
    nlpd = -np.mean(logsumexp(log_density_draws, axis=0) - math.log(len(sigma_draws)))

    sigma2_mean = float(np.mean(sigma_draws**2))
    true_sigma2 = noise_sd_true**2

    epistemic_variance = float(np.mean(np.var(latent_draws, axis=0)))
    aleatoric_variance = sigma2_mean
    predictive_variance = epistemic_variance + aleatoric_variance
    latent_crps = float(np.mean(_crps_ensemble(latent_draws, f_true)))

    negative = np.minimum(derivative_draws, 0.0)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "predictive_coverage": float(np.mean((y_eval >= pred_low) & (y_eval <= pred_high))),
        "predictive_interval_width": float(np.mean(pred_high - pred_low)),
        "nlpd": float(nlpd),
        "latent_coverage": float(np.mean((f_true >= latent_low) & (f_true <= latent_high))),
        "latent_interval_width": float(np.mean(latent_high - latent_low)),
        "latent_crps": latent_crps,
        "epistemic_variance": epistemic_variance,
        "aleatoric_variance": aleatoric_variance, # aleatoric variance is completely same as sigma2_estimate, but keep both for clarity
        "predictive_variance": predictive_variance,
        "sigma2_estimate": sigma2_mean,
        "sigma2_bias": sigma2_mean - true_sigma2,
        "sigma2_squared_error": (sigma2_mean - true_sigma2) ** 2,
        "derivative_violation_rate": float(np.mean(derivative_draws < -1e-8)),
        "derivative_violation_magnitude": float(np.mean(np.abs(negative))),
    }

# Compute MCMC diagnostics for a given MCMC object, including R-hat, effective sample size, divergences, and tree depth hit rate.
def mcmc_diagnostics(mcmc: MCMC, max_tree_depth: int) -> dict[str, float]:
    grouped = mcmc.get_samples(group_by_chain=True)
    stats = diagnostic_summary(grouped, group_by_chain=True)

    rhat_values = np.concatenate([
        np.asarray(value["r_hat"], dtype=float).ravel()
        for value in stats.values()
    ])
    ess_values = np.concatenate([
        np.asarray(value["n_eff"], dtype=float).ravel()
        for value in stats.values()
    ])

    # R-hat is undefined for one chain, so retain only finite values.
    finite_rhat = rhat_values[np.isfinite(rhat_values)]
    finite_ess = ess_values[np.isfinite(ess_values)]

    # Report the worst diagnostic values across all parameter elements.
    rhat_max = float(np.max(finite_rhat)) if finite_rhat.size else np.nan
    ess_min = float(np.min(finite_ess)) if finite_ess.size else np.nan

    extra = mcmc.get_extra_fields(group_by_chain=True)

    # Count post-warmup divergent transitions.
    divergences = float(np.asarray(extra["diverging"]).sum())

    # Calculate the proportion of draws that reached the tree-depth limit.
    num_steps = np.asarray(extra.get("num_steps", [np.nan]), dtype=float)
    maximum_num_steps = 2**max_tree_depth - 1
    tree_depth_hit_rate = float(np.mean(num_steps >= maximum_num_steps))

    return {
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
        "tree_depth_hit_rate": tree_depth_hit_rate,
    }

# Summarize metrics across replications, computing means, standard deviations, and Monte Carlo standard errors for each metric.
def summarize_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "model", "lambda", "scenario", "sample_size_level", "n_train",
        "noise_level", "noise_sd", "region",
    ]
    metric_cols = [c for c in raw.columns if c not in id_cols + ["replication"]]
    output_metric_cols = [c for c in metric_cols if c != "sigma2_squared_error"]
    rows = []

    for keys, group in raw.groupby(id_cols, dropna=False, sort=False):
        row = dict(zip(id_cols, keys))
        row["n_replications"] = group["replication"].nunique()

        for metric in output_metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            mean = values.mean() if n else np.nan
            sd = values.std(ddof=1) if n > 1 else np.nan
            mcse = sd / np.sqrt(n) if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_mcse"] = mcse

        if "sigma2_squared_error" in group.columns:
            squared_errors = pd.to_numeric(
                group["sigma2_squared_error"], errors="coerce"
            ).dropna()
            n = len(squared_errors)
            mse = squared_errors.mean() if n else np.nan
            rmse = np.sqrt(mse) if n else np.nan
            mse_sd = squared_errors.std(ddof=1) if n > 1 else np.nan
            rmse_sd = (
                mse_sd / (2.0 * rmse)
                if n > 1 and np.isfinite(rmse) and rmse > 0
                else np.nan
            )
            rmse_mcse = rmse_sd / np.sqrt(n) if n > 1 else np.nan
            row["sigma2_rmse_mean"] = rmse
            row["sigma2_rmse_sd"] = rmse_sd
            row["sigma2_rmse_mcse"] = rmse_mcse

        rows.append(row)

    result = pd.DataFrame(rows)

    sigma_rmse_cols = [
        "sigma2_rmse_mean",
        "sigma2_rmse_sd",
        "sigma2_rmse_mcse",
    ]
    insert_after = "sigma2_bias_mcse"
    cols = result.columns.tolist()
    for col in sigma_rmse_cols:
        cols.remove(col)
    insert_pos = cols.index(insert_after) + 1
    cols[insert_pos:insert_pos] = sigma_rmse_cols

    return result[cols]
