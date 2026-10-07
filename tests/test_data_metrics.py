import unittest

import numpy as np
import pandas as pd

from src.data import Standardizer, evaluation_grid, true_function
from src.metrics import summarize_metrics

# Unit tests for the data and metrics modules, ensuring correct functionality of true function evaluation, standardization, grid generation, and metric summarization.
class TestDataAndMetrics(unittest.TestCase):
    def test_correct_function_is_monotone(self):
        x1 = np.linspace(-2, 2, 1001)
        x = np.column_stack([x1, np.zeros_like(x1)])
        self.assertTrue(np.all(np.diff(true_function(x, "correct")) > 0))

    def test_misspecified_function_has_negative_derivative_region(self):
        x1 = np.linspace(-2, 2, 1001)
        derivative = 1 + 0.5 * np.pi * np.cos(np.pi * x1)
        self.assertTrue(np.any(derivative < 0))
        self.assertTrue(np.any(derivative > 0))

    def test_standardization_uses_training_statistics(self):
        x = np.array([[0., 1.], [1., 3.], [2., 5.]])
        y = np.array([2., 4., 6.])
        s = Standardizer.fit(x, y)
        np.testing.assert_allclose(s.transform_x(x).mean(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(float(s.transform_y(y).mean()), 0.0)

    def test_grid_has_both_regions(self):
        _, regions = evaluation_grid((-2, 2), 21, (-1, 1))
        self.assertIn("interpolation", regions)
        self.assertIn("extrapolation", regions)

    def test_mcse(self):
        base = dict(model="naive", **{"lambda": np.nan}, scenario="correct",
                    sample_size_level="small", n_train=10, noise_level="low",
                    noise_sd=.2, region="interpolation")
        raw = pd.DataFrame([{**base, "replication": i, "rmse": v}
                            for i, v in enumerate([1., 2., 3.])])
        result = summarize_metrics(raw)
        self.assertAlmostEqual(result.loc[0, "mean"], 2.0)
        self.assertAlmostEqual(result.loc[0, "mcse"], 1 / np.sqrt(3))


if __name__ == "__main__":
    unittest.main()

