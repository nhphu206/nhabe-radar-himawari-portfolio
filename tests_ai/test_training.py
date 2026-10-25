"""Training primitive tests on tiny synthetic tensors."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.model import SmallUNet2D
from nhabe_forecasting.training import (
    TRAINING_CONFIG, evaluate_forecast, evaluate_normalized_mse, make_optimizer, overfit_gate,
    load_model_checkpoint, save_model_checkpoint, seed_everything, train_one_epoch,
)


class TrainingTests(unittest.TestCase):
    def test_repository_config_matches_training_contract(self) -> None:
        config = yaml.safe_load((ROOT / "configs/training.yaml").read_text())
        self.assertEqual(config, TRAINING_CONFIG)

    def test_one_epoch_updates_parameters_and_is_finite(self) -> None:
        seed_everything(2025)
        samples = []
        for _ in range(2):
            inputs = torch.randn(2, 16, 16)
            target = inputs[:1].clone()
            samples.append({"inputs": inputs, "target": target,
                            "target_mask": torch.ones_like(target, dtype=torch.bool)})
        loader = DataLoader(samples, batch_size=1, shuffle=False)
        model = SmallUNet2D(2)
        optimizer = make_optimizer(model, TRAINING_CONFIG)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        initial = evaluate_normalized_mse(model, loader, torch.device("cpu"))
        updated = train_one_epoch(model, loader, optimizer, torch.device("cpu"), 1.0)
        final = evaluate_normalized_mse(model, loader, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor([initial, updated, final])).all())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_unknown_optimizer_is_rejected(self) -> None:
        model = SmallUNet2D(2)
        config = dict(TRAINING_CONFIG); config["optimizer"] = "unknown"
        with self.assertRaises(ValueError):
            make_optimizer(model, config)

    def test_overfit_gate_uses_documented_final_loss_ratio(self) -> None:
        passed, ratio = overfit_gate(10.0, 4.0, 0.5)
        self.assertTrue(passed)
        self.assertEqual(ratio, 0.4)
        passed, _ = overfit_gate(10.0, 5.1, 0.5)
        self.assertFalse(passed)
        with self.assertRaises(ValueError):
            overfit_gate(float("nan"), 1.0, 0.5)

    def test_forecast_evaluation_uses_common_mask_and_dbz_units(self) -> None:
        class FirstChannel(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, :1]
        sample = {
            "inputs": torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                     [[1.0, 1.0], [1.0, 0.0]]]),
            "target": torch.tensor([[[2.0, 2.0], [1.0, 9.0]]]),
            "target_mask": torch.tensor([[[True, True], [True, True]]]),
            "input_radar_mask": torch.tensor([[[True, True], [True, False]]]),
        }
        result = evaluate_forecast(FirstChannel(), DataLoader([sample], batch_size=1),
                                   torch.device("cpu"), radar_std_dbz=2.0)
        self.assertEqual(result["common_valid_pixels"], 3)
        self.assertEqual(result["target_valid_pixels"], 4)
        self.assertAlmostEqual(result["mae_dbz_common_mask"], 2.0)
        self.assertAlmostEqual(result["rmse_dbz_common_mask"], (20 / 3) ** 0.5)

    def test_checkpoint_reopens_with_weights_only_loader(self) -> None:
        model = SmallUNet2D(2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_model_checkpoint(path, model, "radar", 3, {"rmse": 1.5},
                                  {"torch_version": str(torch.__version__)})
            document = load_model_checkpoint(path)
        self.assertEqual(document["epoch"], 3)
        self.assertEqual(document["input_mode"], "radar")
        self.assertEqual(set(document["model_state_dict"]), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
