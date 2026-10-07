from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Define a normal weight prior with a specified scale.
def _normal_weight(name: str, shape: tuple[int, ...], c: float, d_in: int):
    scale = c / jnp.sqrt(float(d_in))
    return numpyro.sample(name, dist.Normal(0.0, scale).expand(shape).to_event(len(shape)))

# Define a positive weight prior using a half-normal distribution for Hard monotonicity constraints.
def _positive_weight(name: str, shape: tuple[int, ...], c: float, d_in: int):
    """Positive half-normal prior; support is handled directly by NUTS."""
    scale = c / jnp.sqrt(float(d_in))
    return numpyro.sample(name, dist.HalfNormal(scale).expand(shape).to_event(len(shape)))

# Define the forward pass for Naive and Soft constrained neural network model.
def standard_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    h1 = jnp.tanh(x @ params["w1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    return (h2 @ params["w3"] + params["b3"]).squeeze(-1)

# Define the forward pass for a hard monotonicity constrained neural network model.
def hard_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    # The x1 row and all downstream weights are positive. Since tanh is
    # increasing, every path derivative from x1 to the output is non-negative.
    # The x2 row is unconstrained, so no monotonicity direction is imposed on x2.
    w1 = jnp.concatenate([params["w1_x1"], params["w1_x2"]], axis=0)
    h1 = jnp.tanh(x @ w1 + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    return (h2 @ params["w3"] + params["b3"]).squeeze(-1)

# Define the Bayesian neural network model for Naive and Soft constrained models.
def _standard_parameters(hidden_dim: int, prior_c: float, bias_scale: float):
    h = hidden_dim
    return {
        "w1": _normal_weight("w1", (2, h), prior_c, 2),
        "b1": numpyro.sample("b1", dist.Normal(0, bias_scale).expand((h,)).to_event(1)),
        "w2": _normal_weight("w2", (h, h), prior_c, h),
        "b2": numpyro.sample("b2", dist.Normal(0, bias_scale).expand((h,)).to_event(1)),
        "w3": _normal_weight("w3", (h, 1), prior_c, h),
        "b3": numpyro.sample("b3", dist.Normal(0, bias_scale)),
    }

# Define the Bayesian neural network model for Hard monotonicity constrained models.
def _hard_parameters(hidden_dim: int, prior_c: float, bias_scale: float):
    h = hidden_dim
    return {
        "w1_x1": _positive_weight("w1_x1", (1, h), prior_c, 2),
        "w1_x2": _normal_weight("w1_x2", (1, h), prior_c, 2),
        "b1": numpyro.sample("b1", dist.Normal(0, bias_scale).expand((h,)).to_event(1)),
        "w2": _positive_weight("w2", (h, h), prior_c, h),
        "b2": numpyro.sample("b2", dist.Normal(0, bias_scale).expand((h,)).to_event(1)),
        "w3": _positive_weight("w3", (h, 1), prior_c, h),
        "b3": numpyro.sample("b3", dist.Normal(0, bias_scale)),
    }

# Define the Bayesian neural network model, which can be either Naive, Soft, or Hard constrained based on the specified model kind.
def bnn_model(
    x: jnp.ndarray,
    y: jnp.ndarray | None = None,
    *,
    model_kind: str,
    hidden_dim: int,
    prior_c: float,
    bias_scale: float,
    constraint_x: jnp.ndarray | None = None,
    penalty_lambda: float = 0.0,
    soft_tau: float = 0.1,
):
    if model_kind == "hard":
        params = _hard_parameters(hidden_dim, prior_c, bias_scale)
        mean = hard_forward(params, x)
    else:
        params = _standard_parameters(hidden_dim, prior_c, bias_scale)
        mean = standard_forward(params, x)
        if model_kind == "soft":
            if constraint_x is None:
                raise ValueError("Soft model requires constraint points.")

            def scalar_f(x_single):
                return standard_forward(params, x_single[None, :])[0]

            gradients = jax.vmap(jax.grad(scalar_f))(constraint_x)[:, 0]
            violation = jax.nn.softplus(-gradients / soft_tau) * soft_tau
            numpyro.factor("monotonicity_penalty", -penalty_lambda * jnp.mean(violation**2))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
    numpyro.deterministic("f", mean)
    with numpyro.plate("observations", x.shape[0]):
        numpyro.sample("y", dist.Normal(mean, sigma_obs), obs=y)

# Return the appropriate forward function based on the model kind.
def forward_for_kind(model_kind: str):
    return hard_forward if model_kind == "hard" else standard_forward

# Compute the posterior latent values for a given set of samples and input points.
def posterior_latent(samples: dict, x: jnp.ndarray, model_kind: str) -> jnp.ndarray:
    forward = forward_for_kind(model_kind)
    parameter_names = [k for k in samples if k != "sigma_obs"]
    params = {k: samples[k] for k in parameter_names}
    return jax.vmap(lambda p: forward(p, x))(params)

# Compute the posterior derivatives for a given set of samples and input points.
def posterior_derivatives(samples: dict, x: jnp.ndarray, model_kind: str, chunk_size: int = 250) -> np.ndarray:
    forward = forward_for_kind(model_kind)
    parameter_names = [k for k in samples if k != "sigma_obs"]
    n_draws = samples[parameter_names[0]].shape[0]

    def derivative_for_sample(p):
        scalar_f = lambda z: forward(p, z[None, :])[0]
        return jax.vmap(jax.grad(scalar_f))(x)[:, 0]

    derivatives = []
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        params_chunk = {k: samples[k][start:stop] for k in parameter_names}
        chunk = np.asarray(jax.vmap(derivative_for_sample)(params_chunk))
        derivatives.append(chunk)

    return np.concatenate(derivatives, axis=0)