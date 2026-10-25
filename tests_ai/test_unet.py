"""Small U-Net shape and gradient contract tests."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.evaluation import masked_mse
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D, trainable_parameter_count


class SmallUNetTests(unittest.TestCase):
    def test_repository_config_matches_implemented_architecture(self) -> None:
        import yaml
        config = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
        self.assertEqual(config, MODEL_CONFIG)

    def test_radar_and_multimodal_output_shape(self) -> None:
        for channels, expected_parameters in ((2, 482_593), (8, 483_457)):
            model = SmallUNet2D(channels, base_channels=16)
            output = model(torch.zeros(1, channels, 32, 40))
            self.assertEqual(output.shape, (1, 1, 32, 40))
            self.assertEqual(trainable_parameter_count(model), expected_parameters)

    def test_output_is_unrestricted_regression(self) -> None:
        model = SmallUNet2D(2)
        self.assertIsInstance(model.output, torch.nn.Conv2d)
        self.assertFalse(any(isinstance(module, (torch.nn.Sigmoid, torch.nn.Softmax))
                             for module in model.modules()))

    def test_masked_backward_has_finite_gradients(self) -> None:
        model = SmallUNet2D(2)
        prediction = model(torch.randn(1, 2, 32, 32))
        target = torch.randn_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        mask[..., 0, 0] = False
        loss = masked_mse(prediction, target, mask)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_channels_and_spatial_shape_are_rejected(self) -> None:
        model = SmallUNet2D(8)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, 32, 32))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 8, 30, 32))
        with self.assertRaises(ValueError):
            SmallUNet2D(8, base_channels=10)


if __name__ == "__main__":
    unittest.main()
