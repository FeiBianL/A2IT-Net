from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from a2it_net.data import (
    ImageTabularDataset,
    TabularPreprocessor,
    apply_label_mapping,
    build_image_transforms,
    make_paper_protocol_splits,
    predefined_protocol_splits,
)
from a2it_net.models import A2ITNet
from a2it_net.training import CrossValidationTrainer, TrainingSettings
from a2it_net.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A2IT-Net with five-fold validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metadata(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    return pd.read_csv(source)


def build_loader(
    frame: pd.DataFrame,
    indices: np.ndarray,
    numerical: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    data_config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    image_size: int,
    train: bool,
) -> DataLoader:
    selected = frame.iloc[indices]
    dataset = ImageTabularDataset(
        image_values=selected[data_config["image_column"]].astype(str).tolist(),
        numerical=numerical,
        categorical=categorical,
        labels=labels[indices],
        image_root=data_config["image_root"],
        image_extension=data_config.get("image_extension", ""),
        transform=build_image_transforms(train=train, image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def validate_config(config: dict[str, Any], frame: pd.DataFrame) -> None:
    data_config = config["data"]
    model_config = config["model"]
    required_columns = {
        data_config["image_column"],
        data_config["label_column"],
        *data_config["numerical_columns"],
        *data_config["categorical_columns"],
    }
    patient_column = data_config.get("patient_id_column")
    if patient_column:
        required_columns.add(patient_column)
    fold_column = data_config.get("fold_column")
    split_column = data_config.get("split_column")
    if bool(fold_column) != bool(split_column):
        raise ValueError("fold_column and split_column must be configured together")
    if fold_column and split_column:
        required_columns.update([fold_column, split_column])
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    if int(model_config["num_classes"]) < 2:
        raise ValueError("num_classes must be at least 2")
    if set(data_config["numerical_columns"]).intersection(data_config["categorical_columns"]):
        raise ValueError("Numerical and categorical column lists must not overlap")


def build_protocol(
    frame: pd.DataFrame,
    labels: np.ndarray,
    data_config: dict[str, Any],
    seed: int,
):
    fold_column = data_config.get("fold_column")
    split_column = data_config.get("split_column")
    if fold_column and split_column:
        return predefined_protocol_splits(
            frame=frame,
            fold_column=fold_column,
            split_column=split_column,
            train_value=str(data_config.get("train_split_value", "train")),
            validation_value=str(data_config.get("validation_split_value", "val")),
            test_value=str(data_config.get("test_split_value", "test")),
        )
    group_column = data_config.get("patient_id_column")
    groups = frame[group_column].to_numpy() if group_column else None
    return make_paper_protocol_splits(labels=labels, groups=groups, seed=seed)


def aggregate_fold_metrics(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}
    metric_names = summaries[0]["best_validation"].keys()
    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [summary["best_validation"][metric_name] for summary in summaries],
            dtype=np.float64,
        )
        aggregate[metric_name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1 if len(values) > 1 else 0)),
        }
    return aggregate


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    seed = int(training_config.get("seed", 42))
    set_random_seed(seed)
    frame = read_metadata(data_config["metadata_path"])
    validate_config(config, frame)
    labels = apply_label_mapping(frame[data_config["label_column"]], data_config.get("label_mapping"))
    observed_classes = np.unique(labels)
    expected_classes = np.arange(int(model_config["num_classes"]))
    if not np.array_equal(observed_classes, expected_classes):
        raise ValueError(
            f"Encoded labels {observed_classes.tolist()} do not match configured classes "
            f"{expected_classes.tolist()}"
        )
    protocol = build_protocol(frame, labels, data_config, seed)
    folds = range(5) if arguments.fold is None else [arguments.fold]
    device_name = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = TrainingSettings(
        epochs=int(training_config.get("epochs", 200)),
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
        warmup_fraction=float(training_config.get("warmup_fraction", 0.1)),
        gradient_clip_norm=float(training_config.get("gradient_clip_norm", 1.0)),
        early_stopping_patience=int(training_config.get("early_stopping_patience", 50)),
        selection_metric=str(training_config.get("selection_metric", "accuracy")),
        use_amp=bool(training_config.get("use_amp", True)),
    )
    summaries: list[dict[str, Any]] = []
    for fold_index in folds:
        train_indices, validation_indices = protocol.folds[fold_index]
        fold_directory = arguments.output_dir / f"fold_{fold_index}"
        preprocessor = TabularPreprocessor(
            numerical_columns=data_config["numerical_columns"],
            categorical_columns=data_config["categorical_columns"],
            standardize_numerical=bool(data_config.get("standardize_numerical", True)),
        )
        train_numerical, train_categorical = preprocessor.fit_transform(
            frame.iloc[train_indices]
        )
        validation_numerical, validation_categorical = preprocessor.transform(
            frame.iloc[validation_indices]
        )
        preprocessor.save(fold_directory / "preprocessor.json")
        train_loader = build_loader(
            frame,
            train_indices,
            train_numerical,
            train_categorical,
            labels,
            data_config,
            batch_size=int(training_config.get("batch_size", 16)),
            num_workers=int(training_config.get("num_workers", 4)),
            image_size=int(model_config.get("image_size", 224)),
            train=True,
        )
        validation_loader = build_loader(
            frame,
            validation_indices,
            validation_numerical,
            validation_categorical,
            labels,
            data_config,
            batch_size=int(training_config.get("batch_size", 16)),
            num_workers=int(training_config.get("num_workers", 4)),
            image_size=int(model_config.get("image_size", 224)),
            train=False,
        )
        model = A2ITNet(
            num_numerical=len(data_config["numerical_columns"]),
            categorical_cardinalities=preprocessor.categorical_cardinalities,
            num_classes=int(model_config["num_classes"]),
            attribute_dim=int(model_config.get("attribute_dim", 64)),
            fusion_dim=int(model_config.get("fusion_dim", 512)),
            dropout=float(model_config.get("dropout", 0.3)),
            pretrained_image_encoder=not arguments.no_pretrained,
        )
        trainer = CrossValidationTrainer(
            model=model,
            device=device,
            settings=settings,
            output_directory=fold_directory,
        )
        result = trainer.fit(train_loader, validation_loader)
        result["fold"] = fold_index
        summaries.append(result)
    summary_payload = {
        "folds": summaries,
        "aggregate": aggregate_fold_metrics(summaries),
    }
    (arguments.output_dir / "validation_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if len(protocol.heldout_test_indices) > 0:
        heldout_manifest = frame.iloc[protocol.heldout_test_indices][
            [data_config["image_column"], data_config["label_column"]]
        ].drop_duplicates()
        heldout_manifest.to_csv(arguments.output_dir / "heldout_test_manifest.csv", index=False)


if __name__ == "__main__":
    main()
