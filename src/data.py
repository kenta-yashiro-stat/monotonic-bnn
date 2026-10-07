from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Define the true underlying function based on the specified scenario.
def true_function(x: np.ndarray, scenario: str) -> np.ndarray:
    x1, x2 = x[..., 0], x[..., 1]
    if scenario == "correct":
        return 0.7 * x1 + 0.15 * x1**3 + 0.5 * np.sin(np.pi * x2)
    if scenario == "partially_misspecified":
        return x1 + 0.5 * np.sin(np.pi * x1) + 0.5 * np.sin(np.pi * x2)
    if scenario == "misspecified":
        return -0.7 * x1 - 0.15 * x1**3 + 0.5 * np.sin(np.pi * x2)
    raise ValueError(f"Unknown scenario: {scenario}")

# Standardizer for normalizing input and output variables.
@dataclass(frozen=True)
class Standardizer:
    x_mean: np.ndarray
    x_sd: np.ndarray
    y_mean: float
    y_sd: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Standardizer":
        x_sd = x.std(axis=0, ddof=1)
        y_sd = float(y.std(ddof=1))
        if np.any(x_sd <= 0) or y_sd <= 0:
            raise ValueError("Training variables must have positive sample SDs.")
        return cls(x.mean(axis=0), x_sd, float(y.mean()), y_sd)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_sd

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_sd

# Generate a synthetic dataset based on the scenario, sample size, and noise level.
def generate_dataset(
    rng: np.random.Generator,
    n: int,
    noise_sd: float,
    scenario: str,
    train_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(train_bounds[0], train_bounds[1], size=(n, 2))
    f = true_function(x, scenario)
    y = f + rng.normal(0.0, noise_sd, size=n)
    return x, y, f

# Create a grid of evaluation points and classify them as interpolation or extrapolation based on training bounds.
def evaluation_grid(
    bounds: tuple[float, float], grid_size: int, train_bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(bounds[0], bounds[1], grid_size)
    a, b = np.meshgrid(axis, axis, indexing="xy")
    x = np.column_stack([a.ravel(), b.ravel()])
    inside = np.all((x >= train_bounds[0]) & (x <= train_bounds[1]), axis=1)
    region = np.where(inside, "interpolation", "extrapolation")
    return x, region

