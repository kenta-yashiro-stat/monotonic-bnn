from __future__ import annotations

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" # Force JAX to use 4 CPU devices for parallel chains.
os.environ["JAX_COMPILATION_CACHE_DIR"] = r"C:\jax_cache" # Set a custom cache directory for JAX compilation artifacts.

import gc # Import garbage collection module to manage memory usage.

import itertools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from numpyro.infer import MCMC, NUTS

from .data import Standardizer, evaluation_grid, generate_dataset, true_function
from .metrics import mcmc_diagnostics, evaluate_region, summarize_metrics
from .models import bnn_model, posterior_derivatives, posterior_latent

# Run the Bayesian neural network experiments based on the provided configuration.
def model_specs(soft_lambdas: list[float]):
    yield "naive", "naive", np.nan
    for value in soft_lambdas:
        yield f"soft_{value:g}", "soft", float(value)
    yield "hard", "hard", np.nan

# Generate a grid of constraint points for the soft monotonicity constraint.
def constraint_grid(x_std: np.ndarray, size: int) -> np.ndarray:
    lo, hi = x_std.min(axis=0), x_std.max(axis=0)
    a = np.linspace(lo[0], hi[0], size)
    b = np.linspace(lo[1], hi[1], size)
    aa, bb = np.meshgrid(a, b, indexing="xy")
    return np.column_stack([aa.ravel(), bb.ravel()])

# Save the data for the visualization for a specific replication and experimental condition to a compressed NPZ file. But this wan't used for my thesis.
def _save_visualization_npz(
    config: dict,
    rep: int,
    spec_name: str,
    penalty_lambda: float,
    scenario: str,
    size_label: str,
    n: int,
    noise_label: str,
    noise_sd: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    f_eval: np.ndarray,
    latent: np.ndarray,
    sigma: np.ndarray,
):
    if rep >= int(config.get("visualization_replications", 3)):
        return

    rep_dir = Path(config["output_dir"]) / "visualization" / f"rep_{rep:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{scenario}_{size_label}_{noise_label}_{spec_name}.npz"

    payload = {
        "replication": np.asarray(rep),
        "model": np.asarray(spec_name),
        "lambda": np.asarray(penalty_lambda),
        "scenario": np.asarray(scenario),
        "sample_size_level": np.asarray(size_label),
        "n_train": np.asarray(int(n)),
        "noise_level": np.asarray(noise_label),
        "noise_sd": np.asarray(float(noise_sd)),
        "x_train": np.asarray(x_train, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.float32),
        "x_eval": np.asarray(x_eval, dtype=np.float32),
        "f_true": np.asarray(f_eval, dtype=np.float32),
        "latent_draws": np.asarray(latent, dtype=np.float32),
        "sigma_draws": np.asarray(sigma, dtype=np.float32),
    }

    np.savez_compressed(rep_dir / filename, **payload)

# Run the Bayesian neural network experiments based on the provided configuration.
def run(config: dict, filters: dict[str, set[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    filters = filters or {}
    data_cfg, model_cfg, mcmc_cfg = config["data"], config["model"], config["mcmc"]
    train_bounds = tuple(map(float, data_cfg["train_bounds"]))
    eval_bounds = tuple(map(float, data_cfg["evaluation_bounds"]))
    x_eval, regions = evaluation_grid(eval_bounds, int(data_cfg["grid_size"]), train_bounds)
    rows: list[dict] = []
    master = np.random.SeedSequence(int(config["seed"]))
    replication_seeds = master.spawn(int(config["replications"]))

    specs = list(model_specs(model_cfg["soft_lambdas"]))
    for rep, rep_seed in enumerate(replication_seeds):
        condition_seeds = rep_seed.spawn(
            len(data_cfg["scenarios"]) * len(data_cfg["sample_sizes"]) * len(data_cfg["noise_levels"])
        )
        conditions = itertools.product(
            data_cfg["scenarios"], data_cfg["sample_sizes"].items(), data_cfg["noise_levels"].items()
        )
        for condition_seed, (scenario, (size_label, n), (noise_label, noise_sd)) in zip(condition_seeds, conditions):
            if filters.get("scenarios") and scenario not in filters["scenarios"]: continue
            if filters.get("sample_sizes") and size_label not in filters["sample_sizes"]: continue
            if filters.get("noise_levels") and noise_label not in filters["noise_levels"]: continue
            rng = np.random.default_rng(condition_seed)
            x_train, y_train, _ = generate_dataset(rng, int(n), float(noise_sd), scenario, train_bounds)
            scaler = Standardizer.fit(x_train, y_train)
            x_train_std = scaler.transform_x(x_train)
            y_train_std = scaler.transform_y(y_train)
            x_eval_std = scaler.transform_x(x_eval)
            c_grid = constraint_grid(x_train_std, int(model_cfg["constraint_grid_size"]))
            f_eval = true_function(x_eval, scenario)

            for spec_name, kind, penalty_lambda in specs:
                if filters.get("models") and spec_name not in filters["models"]: continue
                print(f"rep={rep:03d} scenario={scenario} n={n} noise={noise_sd} model={spec_name}", flush=True)
                kernel = NUTS(
                    bnn_model,
                    target_accept_prob=float(mcmc_cfg["target_accept_prob"]),
                    max_tree_depth=int(mcmc_cfg["max_tree_depth"]),
                )
                mcmc = MCMC(
                    kernel, num_warmup=int(mcmc_cfg["warmup"]),
                    num_samples=int(mcmc_cfg["samples"]), num_chains=int(mcmc_cfg["chains"]),
                    chain_method="parallel" if jax.local_device_count() >= int(mcmc_cfg["chains"]) else "sequential",
                    progress_bar=True,
                )
                key_seed = int(rng.integers(0, 2**31 - 1))
                started = time.perf_counter()
                mcmc.run(
                    jax.random.PRNGKey(key_seed), jnp.asarray(x_train_std), jnp.asarray(y_train_std),
                    model_kind=kind, hidden_dim=int(model_cfg["hidden_dim"]),
                    prior_c=float(model_cfg["prior_c"]), bias_scale=float(model_cfg["bias_scale"]),
                    constraint_x=jnp.asarray(c_grid) if kind == "soft" else None,
                    penalty_lambda=0.0 if np.isnan(penalty_lambda) else penalty_lambda,
                    soft_tau=float(model_cfg["soft_tau"]),
                    extra_fields=("diverging", "num_steps"),
                )

                diagnostics = mcmc_diagnostics(mcmc, int(mcmc_cfg["max_tree_depth"]))

                runtime = time.perf_counter() - started
                samples = mcmc.get_samples(group_by_chain=False)
                latent_std = np.asarray(posterior_latent(samples, jnp.asarray(x_eval_std), kind))
                deriv_std = np.asarray(posterior_derivatives(samples, jnp.asarray(x_eval_std), kind))
                latent = scaler.y_mean + scaler.y_sd * latent_std
                derivative = deriv_std * scaler.y_sd / scaler.x_sd[0]
                sigma = np.asarray(samples["sigma_obs"]) * scaler.y_sd
                

                _save_visualization_npz(
                    config, rep, spec_name, penalty_lambda, scenario, size_label, int(n),
                    noise_label, float(noise_sd),
                    x_train, y_train, x_eval, f_eval, latent, sigma,
                )

                for region in ("interpolation", "extrapolation"):
                    mask = regions == region
                    metrics = evaluate_region(
                        latent[:, mask], sigma, derivative[:, mask], f_eval[mask], float(noise_sd), rng
                    )
                    rows.append({
                        "replication": rep, "model": spec_name, "lambda": penalty_lambda,
                        "scenario": scenario, "sample_size_level": size_label, "n_train": int(n),
                        "noise_level": noise_label, "noise_sd": float(noise_sd), "region": region,
                        **metrics, **diagnostics, "runtime_seconds": runtime,
                    })

                del mcmc, kernel, samples, latent_std, deriv_std, latent, derivative, sigma
                gc.collect()

            del rng, x_train, y_train, scaler
            del x_train_std, y_train_std, x_eval_std, c_grid, f_eval
            gc.collect()
            jax.clear_caches() # Clear JAX compilation caches to free up memory and avoid potential memory leaks during repeated runs.

    raw = pd.DataFrame(rows)
    return raw, summarize_metrics(raw)


def save_results(config: dict, raw: pd.DataFrame, summary: pd.DataFrame):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    with (output / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

