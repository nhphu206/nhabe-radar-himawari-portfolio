"""Locked-test metric helpers are tested without opening research test data."""

from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.final_evaluation import evaluate_method, relative_reduction


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "inputs": torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                     [[1.0, 1.0], [1.0, 0.0]]]),
            "target": torch.tensor([[[2.0, 2.0], [1.0, 9.0]]]),
            "target_mask": torch.tensor([[[True, True], [True, True]]]),
            "input_radar_mask": torch.tensor([[[True, True], [True, False]]]),
            "pair_id": "pair", "input_time_utc": "input", "target_time_utc": "target",
        }
        self.loader = DataLoader([self.sample], batch_size=1)

    def test_persistence_known_global_and_per_sample_metrics(self) -> None:
        summary, rows = evaluate_method(
            "persistence", self.loader, torch.device("cpu"), radar_std_dbz=2.0,
        )
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(summary["evaluable_pixels"], 3)
        self.assertAlmostEqual(summary["mae_dbz"], 2.0)
        self.assertAlmostEqual(summary["rmse_dbz"], (20 / 3) ** 0.5)
        self.assertEqual(rows[0]["method"], "persistence")

    def test_model_path_and_relative_reduction(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros_like(inputs[:, :1])
        summary, _ = evaluate_method(
            "model", self.loader, torch.device("cpu"), 1.0, ZeroModel(),
        )
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(relative_reduction(10.0, 8.0), 20.0)
        with self.assertRaises(ValueError):
            relative_reduction(0.0, 1.0)

    def test_method_model_contract_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_method("persistence", self.loader, torch.device("cpu"), 1.0,
                            torch.nn.Identity())
        with self.assertRaises(ValueError):
            evaluate_method("learned", self.loader, torch.device("cpu"), 1.0)


if __name__ == "__main__":
    unittest.main()
