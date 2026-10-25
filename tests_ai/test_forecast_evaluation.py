"""Masked metric tests for forecasting evaluation."""

import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.evaluation import (
    RegressionAccumulator, cpu_float64, masked_mae, masked_mse, masked_rmse,
)


class MaskedMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = torch.tensor([1.0, 2.0, 100.0])
        self.target = torch.tensor([0.0, 4.0, 0.0])
        self.mask = torch.tensor([True, True, False])

    def test_known_mae_mse_and_rmse(self) -> None:
        self.assertAlmostEqual(masked_mae(self.prediction, self.target, self.mask).item(), 1.5)
        self.assertAlmostEqual(masked_mse(self.prediction, self.target, self.mask).item(), 2.5)
        self.assertAlmostEqual(masked_rmse(self.prediction, self.target, self.mask).item(), math.sqrt(2.5))

    def test_masked_mse_keeps_gradient(self) -> None:
        prediction = self.prediction.clone().requires_grad_(True)
        masked_mse(prediction, self.target, self.mask).backward()
        self.assertTrue(torch.equal(prediction.grad, torch.tensor([1.0, -2.0, 0.0])))

    def test_accumulator_is_pixel_weighted_across_batches(self) -> None:
        accumulator = RegressionAccumulator()
        accumulator.update(torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([True]))
        accumulator.update(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([4.0, 4.0, 4.0]),
                           torch.tensor([True, True, True]))
        result = accumulator.result()
        self.assertEqual(result["valid_pixels"], 4)
        self.assertEqual(result["mae_dbz"], 3.5)
        self.assertAlmostEqual(result["rmse_dbz"], math.sqrt(13))

    def test_invalid_shapes_mask_and_values_are_rejected(self) -> None:
        cases = [
            (self.prediction[:2], self.target, self.mask),
            (self.prediction, self.target, self.mask.float()),
            (self.prediction, self.target, torch.zeros(3, dtype=torch.bool)),
            (torch.tensor([float("nan"), 0.0, 0.0]), self.target, self.mask),
        ]
        for prediction, target, mask in cases:
            with self.assertRaises(ValueError):
                masked_mae(prediction, target, mask)

    def test_metric_transfer_finishes_on_cpu_float64(self) -> None:
        result = cpu_float64(torch.tensor([1.25], dtype=torch.float32))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.item(), 1.25)


if __name__ == "__main__":
    unittest.main()
