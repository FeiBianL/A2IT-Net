import unittest

import numpy as np

from a2it_net.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_use_positive_class_and_negative_specificity(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
        metrics = classification_metrics(labels, probabilities, loss=0.5)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["sensitivity"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
