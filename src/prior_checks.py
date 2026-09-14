from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_grid(bounds: tuple[float, float], grid_size: int) -> tuple[jnp.ndarray, np.ndarray]:
    axis = np.linspace(bounds[0], bounds[1], grid_size)
    x1, x2 = np.meshgrid(axis, axis, indexing="xy")
    grid = np.column_stack([x1.ravel(), x2.ravel()])
    return jnp.asarray(grid), axis


def sample_standard_prior(
    key, num_samples: int, hidden_dim: int, c: float, bias_scale: float
) -> dict:
    keys = jax.random.split(key, 6)
    h = hidden_dim

    return {
        "w1": jax.random.normal(keys[0], (num_samples, 2, h)) * c / jnp.sqrt(2.0),
        "b1": jax.random.normal(keys[1], (num_samples, h)) * bias_scale,
        "w2": jax.random.normal(keys[2], (num_samples, h, h)) * c / jnp.sqrt(float(h)),
        "b2": jax.random.normal(keys[3], (num_samples, h)) * bias_scale,
        "w3": jax.random.normal(keys[4], (num_samples, h, 1)) * c / jnp.sqrt(float(h)),
        "b3": jax.random.normal(keys[5], (num_samples,)) * bias_scale,
    }


def sample_hard_prior(
    key, num_samples: int, hidden_dim: int, c: float, bias_scale: float
) -> dict:
    keys = jax.random.split(key, 7)
    h = hidden_dim

    return {
        "w1_x1": jnp.abs(jax.random.normal(keys[0], (num_samples, 1, h))) * c / jnp.sqrt(2.0),
        "w1_x2": jax.random.normal(keys[1], (num_samples, 1, h)) * c / jnp.sqrt(2.0),
        "b1": jax.random.normal(keys[2], (num_samples, h)) * bias_scale,
        "w2": jnp.abs(jax.random.normal(keys[3], (num_samples, h, h))) * c / jnp.sqrt(float(h)),
        "b2": jax.random.normal(keys[4], (num_samples, h)) * bias_scale,
        "w3": jnp.abs(jax.random.normal(keys[5], (num_samples, h, 1))) * c / jnp.sqrt(float(h)),
        "b3": jax.random.normal(keys[6], (num_samples,)) * bias_scale,
    }


def evaluate_standard_sample(params: dict, grid: jnp.ndarray):
    z1 = grid @ params["w1"] + params["b1"]
    h1 = jnp.tanh(z1)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = jnp.tanh(z2)
    output = (h2 @ params["w3"]).squeeze(-1) + params["b3"]

    def scalar_function(x):
        a1 = jnp.tanh(x @ params["w1"] + params["b1"])
        a2 = jnp.tanh(a1 @ params["w2"] + params["b2"])
        return (a2 @ params["w3"]).squeeze() + params["b3"]

    gradient_x1 = jax.vmap(jax.grad(scalar_function))(grid)[:, 0]
    return output, gradient_x1, z1, z2


def evaluate_hard_sample(params: dict, grid: jnp.ndarray):
    w1 = jnp.concatenate([params["w1_x1"], params["w1_x2"]], axis=0)
    z1 = grid @ w1 + params["b1"]
    h1 = jnp.tanh(z1)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = jnp.tanh(z2)
    output = (h2 @ params["w3"]).squeeze(-1) + params["b3"]

    def scalar_function(x):
        a1 = jnp.tanh(x @ w1 + params["b1"])
        a2 = jnp.tanh(a1 @ params["w2"] + params["b2"])
        return (a2 @ params["w3"]).squeeze() + params["b3"]

    gradient_x1 = jax.vmap(jax.grad(scalar_function))(grid)[:, 0]
    return output, gradient_x1, z1, z2


def evaluate_prior(
    params: dict,
    grid: jnp.ndarray,
    model_kind: str,
    saturation_threshold: float,
    interpolation_mask: np.ndarray,
) -> tuple[dict, np.ndarray]:
    if model_kind == "standard":
        evaluate_one = evaluate_standard_sample
    elif model_kind == "hard":
        evaluate_one = evaluate_hard_sample
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    outputs, gradients, z1, z2 = jax.vmap(lambda p: evaluate_one(p, grid))(params)

    outputs_np = np.asarray(outputs)
    gradients_np = np.asarray(gradients)
    z1_np = np.asarray(z1)
    z2_np = np.asarray(z2)

    extrapolation_mask = ~interpolation_mask

    ranges = np.ptp(outputs_np, axis=1)
    standard_deviations = np.std(outputs_np, axis=1, ddof=0)
    negative_gradients = np.minimum(gradients_np, 0.0)

    metrics = {
        "maximum_absolute_output": float(np.max(np.abs(outputs_np))),
        "prior_predictive_variance": float(np.var(outputs_np, ddof=0)),
        "mean_intra_function_range": float(np.mean(ranges)),
        "mean_intra_function_standard_deviation": float(np.mean(standard_deviations)),
        "mean_absolute_gradient_x1": float(np.mean(np.abs(gradients_np))),
        "derivative_violation_rate": float(np.mean(gradients_np < -1e-8)),
        "derivative_violation_magnitude": float(np.mean(np.abs(negative_gradients))),
        "layer1_saturation_rate_interpolation": float(
            np.mean(np.abs(z1_np[:, interpolation_mask, :]) > saturation_threshold)
        ),
        "layer1_saturation_rate_extrapolation": float(
            np.mean(np.abs(z1_np[:, extrapolation_mask, :]) > saturation_threshold)
        ),
        "layer2_saturation_rate_interpolation": float(
            np.mean(np.abs(z2_np[:, interpolation_mask, :]) > saturation_threshold)
        ),
        "layer2_saturation_rate_extrapolation": float(
            np.mean(np.abs(z2_np[:, extrapolation_mask, :]) > saturation_threshold)
        ),
    }

    return metrics, outputs_np


def save_heatmap_panel(
    outputs_by_c: dict[float, np.ndarray],
    axis: np.ndarray,
    c_values: list[float],
    model_kind: str,
    output_path: Path,
    n_functions: int,
):
    grid_size = len(axis)
    n_c = len(c_values)

    for c in c_values:
        if outputs_by_c[c].shape[0] < n_functions:
            raise ValueError(
                f"Not enough prior samples for c={c:g}. "
                f"Requested {n_functions}, but only "
                f"{outputs_by_c[c].shape[0]} are available."
            )

    selected_by_c = {
        c: outputs_by_c[c][:n_functions].reshape(n_functions, grid_size, grid_size)
        for c in c_values
    }

    all_selected = np.concatenate(
        [selected_by_c[c].reshape(n_functions, -1) for c in c_values],
        axis=1,
    )

    color_min = float(np.min(all_selected))
    color_max = float(np.max(all_selected))

    if color_min == color_max:
        color_max = color_min + np.finfo(float).eps

    fig, axes = plt.subplots(
        n_functions,
        n_c,
        figsize=(3.1 * n_c, 3.0 * n_functions),
        constrained_layout=True,
        squeeze=False,
    )

    image = None

    for column, c in enumerate(c_values):
        for row in range(n_functions):
            plot_axis = axes[row, column]

            image = plot_axis.imshow(
                selected_by_c[c][row],
                origin="lower",
                extent=[axis[0], axis[-1], axis[0], axis[-1]],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                aspect="equal",
            )

            if row == 0:
                plot_axis.set_title(rf"$c={c:g}$")

            if row == n_functions - 1:
                plot_axis.set_xlabel(r"$x_1$")

            if column == 0:
                plot_axis.set_ylabel(rf"Draw {row + 1}" "\n" r"$x_2$")

    fig.colorbar(
        image,
        ax=axes,
        label=r"$f(x_1,x_2)$",
        shrink=0.85,
    )

    model_title = "Naive / Soft" if model_kind == "standard" else "Hard-constrained"

    fig.suptitle(
        f"{model_title} prior functions",
        fontsize=15,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_prior_checks(config: dict) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = tuple(map(float, config["input_bounds"]))
    grid, axis = make_grid(bounds, int(config["grid_size"]))

    grid_original = np.asarray(grid)
    interpolation_mask = np.all(
        (grid_original >= -1.0) & (grid_original <= 1.0),
        axis=1,
    )

    training_input_mean = 0.0
    training_input_sd = 1.0 / np.sqrt(3.0)
    grid = (grid - training_input_mean) / training_input_sd

    c_values = [float(c) for c in config["c_values"]]

    combinations = [
        (model_kind, c)
        for model_kind in config["model_kinds"]
        for c in c_values
    ]

    master_key = jax.random.PRNGKey(int(config["seed"]))
    keys = jax.random.split(master_key, len(combinations))

    rows = []

    outputs_by_model = {
        model_kind: {}
        for model_kind in config["model_kinds"]
    }

    for key, (model_kind, c) in zip(keys, combinations):
        if model_kind == "standard":
            sampling_function = sample_standard_prior
        elif model_kind == "hard":
            sampling_function = sample_hard_prior
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")

        params = sampling_function(
            key=key,
            num_samples=int(config["num_samples"]),
            hidden_dim=int(config["hidden_dim"]),
            c=c,
            bias_scale=float(config["bias_scale"]),
        )

        metrics, outputs = evaluate_prior(
            params=params,
            grid=grid,
            model_kind=model_kind,
            saturation_threshold=float(config["saturation_threshold"]),
            interpolation_mask=interpolation_mask,
        )

        outputs_by_model[model_kind][c] = outputs

        rows.append({
            "model_kind": model_kind,
            "c": c,
            "num_samples": int(config["num_samples"]),
            **metrics,
        })

        print(f"model={model_kind}, c={c:g}: evaluated")

    for model_kind in config["model_kinds"]:
        filename = f"prior_heatmaps_{model_kind}.png"

        save_heatmap_panel(
            outputs_by_c=outputs_by_model[model_kind],
            axis=axis,
            c_values=c_values,
            model_kind=model_kind,
            output_path=output_dir / filename,
            n_functions=int(config["heatmap_functions"]),
        )

        print(f"saved {output_dir / filename}")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "prior_check_metrics.csv", index=False)

    return results