#!/usr/bin/env python3
"""Fine-tune 25-omr seg_net for background/notehead/stem/beam segmentation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


PATCH_SIZE = 288
CLASS_TO_MASK = {"notehead": 1, "stem": 2, "beam": 3}


@dataclass(frozen=True)
class Annotation:
    class_id: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Page:
    image_path: Path
    label_path: Path
    work: str
    annotations: tuple[Annotation, ...]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replace the 25-omr seg_net head and fine-tune it for notehead/stem/beam."
    )
    parser.add_argument("dataset", type=Path, help="Dataset root containing images/, labels/, classes.txt")
    parser.add_argument("--base-model-dir", type=Path, default=root / "25-omr/omr/checkpoints/seg_net")
    parser.add_argument("--output-dir", type=Path, default=root / "primitive_training_output")
    parser.add_argument("--train-works", default="Beethoven_Op090,Beethoven_Op101,Beethoven_Op106")
    parser.add_argument("--validation-works", default="Beethoven_Op109")
    parser.add_argument("--notehead-ids", help="Comma-separated YOLO class IDs; auto-detected by class name by default")
    parser.add_argument("--stem-ids", help="Comma-separated YOLO class IDs")
    parser.add_argument("--beam-ids", help="Comma-separated YOLO class IDs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Validate data without importing TensorFlow")
    parser.add_argument(
        "--allow-small-only",
        action="store_true",
        help="Allow training when all mapped classes are *Small; this does not meet full-page primitive detection goals",
    )
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def parse_id_list(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_classes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def auto_class_ids(classes: list[str], prefix: str) -> set[int]:
    return {index for index, name in enumerate(classes) if name.lower().startswith(prefix)}


def resolve_class_map(args: argparse.Namespace, classes: list[str]) -> dict[str, set[int]]:
    result = {
        "notehead": parse_id_list(args.notehead_ids) or auto_class_ids(classes, "notehead"),
        "stem": parse_id_list(args.stem_ids) or auto_class_ids(classes, "stem"),
        "beam": parse_id_list(args.beam_ids) or auto_class_ids(classes, "beam"),
    }
    missing = [name for name, ids in result.items() if not ids]
    if missing:
        raise ValueError(f"No class IDs resolved for: {', '.join(missing)}")
    targets = list(result)
    overlap: set[int] = set()
    for first_index, first in enumerate(targets):
        for second in targets[first_index + 1 :]:
            overlap |= result[first] & result[second]
    if overlap:
        raise ValueError(f"Class IDs may not map to multiple targets: {sorted(overlap)}")
    return result


def work_from_stem(stem: str) -> str:
    return stem.split("-", 1)[0]


def parse_yolo_labels(label_path: Path, width: int, height: int) -> tuple[Annotation, ...]:
    annotations: list[Annotation] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
        x0 = max(0, round((center_x - box_width / 2) * width))
        y0 = max(0, round((center_y - box_height / 2) * height))
        x1 = min(width, round((center_x + box_width / 2) * width))
        y1 = min(height, round((center_y + box_height / 2) * height))
        if x1 > x0 and y1 > y0:
            annotations.append(Annotation(class_id, (x0, y0, x1, y1)))
    return tuple(annotations)


def load_pages(dataset: Path) -> list[Page]:
    image_dir = dataset / "images"
    label_dir = dataset / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Dataset must contain images/ and labels/")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    pages: list[Page] = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        annotations = parse_yolo_labels(label_path, image.shape[1], image.shape[0])
        pages.append(Page(image_path, label_path, work_from_stem(image_path.stem), annotations))
    return pages


def mapped_name(class_id: int, class_map: dict[str, set[int]]) -> str | None:
    for name, ids in class_map.items():
        if class_id in ids:
            return name
    return None


def dataset_report(pages: list[Page], classes: list[str], class_map: dict[str, set[int]]) -> dict:
    counts: Counter[str] = Counter()
    raw_counts: Counter[int] = Counter()
    for page in pages:
        for annotation in page.annotations:
            raw_counts[annotation.class_id] += 1
            target = mapped_name(annotation.class_id, class_map)
            if target:
                counts[target] += 1
    selected_names = {
        target: [classes[index] if index < len(classes) else f"class_{index}" for index in sorted(ids)]
        for target, ids in class_map.items()
    }
    small_only = all(
        names and all("small" in name.lower() for name in names)
        for names in selected_names.values()
    )
    return {
        "pages": len(pages),
        "works": dict(Counter(page.work for page in pages)),
        "selected_class_names": selected_names,
        "target_counts": dict(counts),
        "small_only": small_only,
        "unmapped_annotations": sum(raw_counts.values()) - sum(counts.values()),
    }


def crop_bounds(center_x: float, center_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = min(max(0, round(center_x - PATCH_SIZE / 2)), max(0, width - PATCH_SIZE))
    y0 = min(max(0, round(center_y - PATCH_SIZE / 2)), max(0, height - PATCH_SIZE))
    return x0, y0, min(width, x0 + PATCH_SIZE), min(height, y0 + PATCH_SIZE)


def build_patch_index(pages: Iterable[Page], class_map: dict[str, set[int]]) -> list[tuple[Page, tuple[int, int, int, int]]]:
    index: list[tuple[Page, tuple[int, int, int, int]]] = []
    for page in pages:
        image = cv2.imread(str(page.image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        seen: set[tuple[int, int]] = set()
        for annotation in page.annotations:
            if mapped_name(annotation.class_id, class_map) is None:
                continue
            x0, y0, x1, y1 = annotation.bbox
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            key = (round(center_x / (PATCH_SIZE // 2)), round(center_y / (PATCH_SIZE // 2)))
            if key in seen:
                continue
            seen.add(key)
            index.append((page, crop_bounds(center_x, center_y, image.shape[1], image.shape[0])))
    return index


def ink_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def build_page_mask(image: np.ndarray, page: Page, class_map: dict[str, set[int]]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target = np.zeros(gray.shape, dtype=np.uint8)
    # Beam first, then stem, then notehead. The smaller/more-specific glyph
    # wins where YOLO boxes overlap.
    for primitive in ("beam", "stem", "notehead"):
        for annotation in page.annotations:
            if mapped_name(annotation.class_id, class_map) != primitive:
                continue
            x0, y0, x1, y1 = annotation.bbox
            roi = ink_mask(gray[y0:y1, x0:x1])
            if primitive == "beam":
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
            elif primitive == "stem":
                kernel_height = max(3, (y1 - y0) // 3)
                opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((kernel_height, 1), np.uint8))
                if np.any(opened):
                    roi = opened
            else:
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            region = target[y0:y1, x0:x1]
            region[roi > 0] = CLASS_TO_MASK[primitive]
    return target


def patch_generator(
    patch_index: list[tuple[Page, tuple[int, int, int, int]]],
    class_map: dict[str, set[int]],
    seed: int,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = random.Random(seed)
    order = list(range(len(patch_index)))
    if shuffle:
        rng.shuffle(order)
    loaded_path: Path | None = None
    loaded_image: np.ndarray | None = None
    loaded_mask: np.ndarray | None = None
    class_weights = np.asarray([0.12, 4.0, 3.0, 3.5], dtype=np.float32)
    for item_index in order:
        page, (x0, y0, x1, y1) = patch_index[item_index]
        if loaded_path != page.image_path:
            loaded_image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
            if loaded_image is None:
                raise ValueError(f"Could not read {page.image_path}")
            loaded_mask = build_page_mask(loaded_image, page, class_map)
            loaded_path = page.image_path
        assert loaded_image is not None and loaded_mask is not None
        image_patch = loaded_image[y0:y1, x0:x1]
        mask_patch = loaded_mask[y0:y1, x0:x1]
        if image_patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            image_patch = cv2.copyMakeBorder(image_patch, 0, PATCH_SIZE - image_patch.shape[0], 0, PATCH_SIZE - image_patch.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask_patch = cv2.copyMakeBorder(mask_patch, 0, PATCH_SIZE - mask_patch.shape[0], 0, PATCH_SIZE - mask_patch.shape[1], cv2.BORDER_CONSTANT, value=0)
        sample_weight = class_weights[mask_patch]
        yield image_patch.astype(np.float32), mask_patch.astype(np.int32), sample_weight


def build_model(tf, base_model_dir: Path):
    architecture = (base_model_dir / "arch.json").read_text(encoding="utf-8")
    base = tf.keras.models.model_from_json(architecture)
    base.load_weights(str(base_model_dir / "weights.h5"))
    features = base.layers[-1].input
    output = tf.keras.layers.Conv2D(4, 1, activation="softmax", name="primitive_head")(features)
    model = tf.keras.Model(inputs=base.input, outputs=output, name="primitive_seg_net")
    for layer in base.layers:
        layer.trainable = False
    return model, base


def make_dataset(tf, index, class_map, batch_size, seed, shuffle):
    signature = (
        tf.TensorSpec((PATCH_SIZE, PATCH_SIZE, 3), tf.float32),
        tf.TensorSpec((PATCH_SIZE, PATCH_SIZE), tf.int32),
        tf.TensorSpec((PATCH_SIZE, PATCH_SIZE), tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator(index, class_map, seed, shuffle),
        output_signature=signature,
    )
    if shuffle:
        dataset = dataset.shuffle(min(512, max(16, len(index))), seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    classes = load_classes(dataset / "classes.txt")
    class_map = resolve_class_map(args, classes)
    pages = load_pages(dataset)
    report = dataset_report(pages, classes, class_map)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["small_only"] and not args.allow_small_only:
        print(
            "Refusing to train: every selected target class is *Small. Add complete regular notehead/stem/beam labels, or pass --allow-small-only for an intentional experiment.",
            file=sys.stderr,
        )
        return 2
    train_works = {item.strip() for item in args.train_works.split(",") if item.strip()}
    validation_works = {item.strip() for item in args.validation_works.split(",") if item.strip()}
    if train_works & validation_works:
        raise ValueError("Training and validation works must not overlap")
    train_pages = [page for page in pages if page.work in train_works]
    validation_pages = [page for page in pages if page.work in validation_works]
    train_index = build_patch_index(train_pages, class_map)
    validation_index = build_patch_index(validation_pages, class_map)
    print(f"train_pages={len(train_pages)} train_patches={len(train_index)}")
    print(f"validation_pages={len(validation_pages)} validation_patches={len(validation_index)}")
    if not train_index or not validation_index:
        raise ValueError("Training and validation both need at least one annotated patch")
    if args.dry_run:
        return 0

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed. Run: python -m pip install tensorflow", file=sys.stderr)
        return 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    model, base = build_model(tf, args.base_model_dir)
    train_data = make_dataset(tf, train_index, class_map, args.batch_size, args.seed, True)
    validation_data = make_dataset(tf, validation_index, class_map, args.batch_size, args.seed, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "primitive_seg_net.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")],
    )
    warmup = min(args.warmup_epochs, args.epochs)
    if warmup:
        model.fit(train_data, validation_data=validation_data, epochs=warmup, callbacks=callbacks)
    for layer in base.layers[-36:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate * 0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")],
    )
    if args.epochs > warmup:
        model.fit(
            train_data,
            validation_data=validation_data,
            initial_epoch=warmup,
            epochs=args.epochs,
            callbacks=callbacks,
        )
    model.save(best_path)
    metadata = {
        "class_order": ["background", "notehead", "stem", "beam"],
        "class_map": {name: sorted(ids) for name, ids in class_map.items()},
        "dataset_report": report,
        "patch_size": PATCH_SIZE,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.export_onnx:
        try:
            import tf2onnx
        except ImportError:
            print("tf2onnx is not installed; saved .keras but skipped ONNX export", file=sys.stderr)
        else:
            signature = [tf.TensorSpec((None, PATCH_SIZE, PATCH_SIZE, 3), tf.float32, name="input")]
            tf2onnx.convert.from_keras(model, input_signature=signature, opset=17, output_path=str(args.output_dir / "primitive_seg_net.onnx"))
    print(f"Saved model: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
