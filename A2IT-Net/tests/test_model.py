import unittest

import torch

from a2it_net.models import A2ITNet, AdaTab, DecFusion


class ModelTests(unittest.TestCase):
    def test_adatab_shape_and_attention(self) -> None:
        model = AdaTab(
            num_numerical=2,
            categorical_cardinalities=[4, 6],
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=32,
            dropout=0.0,
        )
        numerical = torch.randn(3, 2)
        categorical = torch.tensor([[0, 1], [2, 3], [1, 5]])
        representation, attention = model(numerical, categorical, return_attention=True)
        self.assertEqual(representation.shape, (3, 16))
        self.assertEqual(attention.shape, (3, 4))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))

    def test_decfusion_shape(self) -> None:
        model = DecFusion(image_dim=32, tabular_dim=16, hidden_dim=8, dropout=0.0)
        output = model(torch.randn(3, 32), torch.randn(3, 16))
        self.assertEqual(output.shape, (3, 24))

    def test_complete_model_shape(self) -> None:
        model = A2ITNet(
            num_numerical=2,
            categorical_cardinalities=[4, 6],
            num_classes=3,
            attribute_dim=16,
            fusion_dim=8,
            dropout=0.0,
            pretrained_image_encoder=False,
        )
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.randn(2, 3, 64, 64),
                torch.randn(2, 2),
                torch.tensor([[0, 1], [2, 3]]),
            )
        self.assertEqual(logits.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()

