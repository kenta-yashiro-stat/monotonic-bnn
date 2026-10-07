import unittest

import numpy as np

from src.prior_checks import make_grid

# Unit tests for the prior checks module, ensuring correct functionality of grid generation for prior predictive checks.
class TestPriorChecks(unittest.TestCase):
    def test_grid_shape(self):
        grid, axis = make_grid((-2.0, 2.0), 11)
        self.assertEqual(grid.shape, (121, 2))
        self.assertEqual(axis.shape, (11,))
        np.testing.assert_allclose(np.asarray(grid).min(axis=0), [-2.0, -2.0])
        np.testing.assert_allclose(np.asarray(grid).max(axis=0), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
