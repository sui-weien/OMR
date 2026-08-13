#!/usr/bin/env python3
"""Measure recall on the sparsely annotated Xia primitive boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate annotated-target recall; Xia labels are sparse, so precision is not reported."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("predictions", type=Path, help="Directory containing detect_primitives JSON files")
    return parser.parse_args()


def box_iou(first: list[int], second: list[int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(first_area + second_area - intersection)


def bbox_smaller_overlap(first: list[int], second: list[int]) -> float:
    """Fraction of the SMALLER box's area that the two boxes share.

    A ground-truth "Small" annotation can mark just one short segment of a
    much longer detected beam run, while a detected beam's own bbox is
    sometimes the smaller one and sits entirely inside a looser annotation
    box. Either way, plain IoU is misleadingly low just because the two
    boxes are different sizes by convention, not because the match is
    wrong -- dividing by whichever box is actually smaller (rather than
    always the ground-truth box) answers "is the smaller one covered",
    which is the question that actually matters for a correct match here.
    """
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(min(first_area, second_area))


def stem_match(gt: list[int], prediction: dict[str, Any]) -> float:
    x, top, _, bottom = prediction["line"]
    x0, y0, x1, y1 = gt
    horizontal_gap = max(x0 - x, x - x1, 0)
    tolerance = max(3.0, (x1 - x0) * 1.5)
    if horizontal_gap > tolerance:
        return 0.0
    overlap = max(0, min(bottom, y1) - max(top, y0))
    vertical_recall = overlap / max(1.0, y1 - y0)
    if vertical_recall < 0.5:
        return 0.0
    return vertical_recall * (1.0 - horizontal_gap / tolerance)


def greedy_match(
    ground_truth: list[Any],
    predictions: list[dict[str, Any]],
    score: Callable[[Any, dict[str, Any]], float],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedily pair each ground-truth item with its best-scoring prediction.

    Returns (ground_truth_index, prediction_index, score) triples so callers
    can inspect the matched prediction, not just count it.
    """
    available = set(range(len(predictions)))
    pairs: list[tuple[int, int, float]] = []
    for target_index, target in enumerate(ground_truth):
        ranked = sorted(
            ((score(target, predictions[index]), index) for index in available),
            reverse=True,
        )
        if ranked and ranked[0][0] >= threshold:
            pairs.append((target_index, ranked[0][1], ranked[0][0]))
            available.remove(ranked[0][1])
    return pairs


def greedy_recall(
    ground_truth: list[Any],
    predictions: list[dict[str, Any]],
    score: Callable[[Any, dict[str, Any]], float],
    threshold: float,
) -> tuple[int, list[float]]:
    pairs = greedy_match(ground_truth, predictions, score, threshold)
    return len(pairs), [item[2] for item in pairs]


def _expected_accidental_type(class_name: str) -> str | None:
    lowered = class_name.lower()
    if "flat" in lowered:
        return "flat"
    if "sharp" in lowered:
        return "sharp"
    if "natural" in lowered:
        return "natural"
    return None


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, list[int]]]:
    result: list[tuple[int, list[int]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, cx_text, cy_text, width_text, height_text = line.split()
        cx, cy, box_width, box_height = map(
            float, (cx_text, cy_text, width_text, height_text)
        )
        result.append(
            (
                int(class_id_text),
                [
                    max(0, round((cx - box_width / 2) * width)),
                    max(0, round((cy - box_height / 2) * height)),
                    min(width, round((cx + box_width / 2) * width)),
                    min(height, round((cy + box_height / 2) * height)),
                ],
            )
        )
    return result


def main() -> int:
    args = parse_args()
    classes = [
        line.strip()
        for line in (args.dataset / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    class_ids = {
        "notehead": {index for index, name in enumerate(classes) if name.lower().startswith("notehead")},
        "stem": {index for index, name in enumerate(classes) if name.lower().startswith("stem")},
        "beam": {index for index, name in enumerate(classes) if name.lower().startswith("beam")},
        "accidental": {index for index, name in enumerate(classes) if name.lower().startswith("accidental")},
    }
    accidental_expected_type = {
        index: _expected_accidental_type(classes[index]) for index in class_ids["accidental"]
    }
    totals = {name: 0 for name in class_ids}
    matches = {name: 0 for name in class_ids}
    qualities: dict[str, list[float]] = {name: [] for name in class_ids}
    accidental_type_correct = 0
    accidental_type_total = 0
    accidental_type_confusion: dict[str, dict[str, int]] = {}
    pages = 0
    for prediction_path in sorted(args.predictions.glob("*.json")):
        image_candidates = list((args.dataset / "images").glob(f"{prediction_path.stem}.*"))
        label_path = args.dataset / "labels" / f"{prediction_path.stem}.txt"
        if not image_candidates or not label_path.exists():
            continue
        image = cv2.imread(str(image_candidates[0]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        labels = read_yolo_boxes(label_path, image.shape[1], image.shape[0])
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        gt = {
            name: [bbox for class_id, bbox in labels if class_id in ids]
            for name, ids in class_ids.items()
            if name != "accidental"
        }
        accidental_gt = [
            (bbox, class_id) for class_id, bbox in labels if class_id in class_ids["accidental"]
        ]
        prediction_groups = {
            "notehead": prediction.get("noteheads", []),
            "stem": prediction.get("stems", []),
            "beam": prediction.get("beams", []),
        }
        scorers = {
            "notehead": lambda target, item: box_iou(target, item["bbox"]),
            "stem": stem_match,
            "beam": lambda target, item: bbox_smaller_overlap(target, item["bbox"]),
        }
        thresholds = {"notehead": 0.2, "stem": 0.5, "beam": 0.6}
        for name in ("notehead", "stem", "beam"):
            matched, scores = greedy_recall(
                gt[name], prediction_groups[name], scorers[name], thresholds[name]
            )
            totals[name] += len(gt[name])
            matches[name] += matched
            qualities[name].extend(scores)

        accidental_predictions = prediction.get("accidentals", [])
        accidental_pairs = greedy_match(
            accidental_gt,
            accidental_predictions,
            lambda target, item: box_iou(target[0], item["bbox"]),
            0.15,
        )
        totals["accidental"] += len(accidental_gt)
        matches["accidental"] += len(accidental_pairs)
        qualities["accidental"].extend(item[2] for item in accidental_pairs)
        for gt_index, pred_index, _score in accidental_pairs:
            expected = accidental_expected_type[accidental_gt[gt_index][1]]
            predicted = accidental_predictions[pred_index].get("type")
            if expected is None:
                continue
            accidental_type_total += 1
            if predicted == expected:
                accidental_type_correct += 1
            accidental_type_confusion.setdefault(expected, {}).setdefault(predicted, 0)
            accidental_type_confusion[expected][predicted] += 1
        pages += 1
    report = {
        "pages": pages,
        "note": "Xia-style annotations are sparse; these are annotated-target recall values, not full-page precision.",
        "targets": {
            name: {
                "annotated": totals[name],
                "matched": matches[name],
                "recall": round(matches[name] / totals[name], 4) if totals[name] else None,
                "mean_match_quality": (
                    round(sum(qualities[name]) / len(qualities[name]), 4)
                    if qualities[name]
                    else None
                ),
            }
            for name in class_ids
        },
        "accidental_type_accuracy": {
            "note": "Computed only over matched (present + located) accidentals, so it is independent of the recall figure above.",
            "correct": accidental_type_correct,
            "total": accidental_type_total,
            "accuracy": (
                round(accidental_type_correct / accidental_type_total, 4)
                if accidental_type_total
                else None
            ),
            "confusion_matrix_expected_to_predicted": accidental_type_confusion,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
