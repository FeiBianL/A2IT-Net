import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from a2it_net.data import (
    TabularPreprocessor,
    make_paper_protocol_splits,
    predefined_protocol_splits,
)


class DataTests(unittest.TestCase):
    def test_preprocessor_uses_explicit_missing_and_unknown_categories(self) -> None:
        train = pd.DataFrame(
            {
                "age": [10.0, np.nan, 30.0],
                "sex": ["F", None, "M"],
            }
        )
        validation = pd.DataFrame({"age": [np.nan], "sex": ["X"]})
        preprocessor = TabularPreprocessor(["age"], ["sex"])
        numerical, categorical = preprocessor.fit_transform(train)
        validation_numerical, validation_categorical = preprocessor.transform(validation)
        self.assertEqual(numerical.shape, (3, 1))
        self.assertEqual(categorical.shape, (3, 1))
        self.assertEqual(validation_numerical.shape, (1, 1))
        self.assertEqual(int(validation_categorical[0, 0]), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.json"
            preprocessor.save(path)
            loaded = TabularPreprocessor.load(path)
            loaded_numerical, loaded_categorical = loaded.transform(validation)
        np.testing.assert_allclose(loaded_numerical, validation_numerical)
        np.testing.assert_array_equal(loaded_categorical, validation_categorical)

    def test_patient_groups_do_not_overlap(self) -> None:
        labels = np.tile(np.array([0, 1]), 30)
        groups = np.repeat(np.arange(30), 2)
        protocol = make_paper_protocol_splits(labels, groups, seed=42)
        heldout_groups = set(groups[protocol.heldout_test_indices])
        development_groups = set(groups[protocol.development_indices])
        self.assertTrue(heldout_groups.isdisjoint(development_groups))
        for train_indices, validation_indices in protocol.folds:
            self.assertTrue(
                set(groups[train_indices]).isdisjoint(set(groups[validation_indices]))
            )

    def test_preprocessed_numerical_values_can_bypass_standardization(self) -> None:
        frame = pd.DataFrame({"value": [10.0, 20.0], "category": [0, 1]})
        preprocessor = TabularPreprocessor(
            ["value"],
            ["category"],
            standardize_numerical=False,
        )
        numerical, _ = preprocessor.fit_transform(frame)
        np.testing.assert_allclose(numerical[:, 0], np.array([10.0, 20.0]))

    def test_predefined_five_fold_protocol(self) -> None:
        rows = []
        for fold in range(5):
            for split, count in [("train", 4), ("val", 2), ("test", 2)]:
                rows.extend({"fold": fold, "split": split} for _ in range(count))
        frame = pd.DataFrame(rows)
        protocol = predefined_protocol_splits(frame, "fold", "split")
        self.assertEqual(len(protocol.folds), 5)
        for train_indices, validation_indices in protocol.folds:
            self.assertEqual(len(train_indices), 4)
            self.assertEqual(len(validation_indices), 2)
            self.assertTrue(set(train_indices).isdisjoint(set(validation_indices)))
        self.assertEqual(len(protocol.heldout_test_indices), 10)


if __name__ == "__main__":
    unittest.main()
