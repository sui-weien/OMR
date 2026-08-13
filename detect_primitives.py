#!/usr/bin/env python3
"""Detect noteheads, stems, and beams in one image or an image directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from primitive_omr.detector import (
    DetectionConfig,
    PrimitiveDetector,
    draw_accidental_overlay,
    draw_beam_overlay,
    draw_overlay,
    draw_stem_overlay,
    load_legacy_segmentation,
)
from primitive_omr.inference import (
    rescale_result,
    resize_for_model,
    run_keras_segmentation,
    run_onnx_segmentation,
)
from primitive_omr.rhythm_classifier import SingleStemRhythmClassifier


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detect noteheads, stems, and beams; write JSON and an overlay PNG."
    )
    parser.add_argument("input", type=Path, help="Image file or directory of images")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "primitive_output"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "25-omr/omr/checkpoints/seg_net/model.onnx",
        help="Original seg_net ONNX or a fine-tuned .onnx/.keras primitive model",
    )
    parser.add_argument(
        "--model-kind",
        choices=("original", "primitive"),
        default="original",
        help="Original channels are background/stem-rest/notehead/clef; primitive channels are background/notehead/stem/beam with optional flag",
    )
    parser.add_argument(
        "--legacy-segmentation",
        type=Path,
        help="Use one existing 25-omr .npy cache instead of running a model (single image only)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore segmentation cache")
    parser.add_argument("--notehead-threshold", type=float, default=0.28)
    parser.add_argument("--stem-threshold", type=float, default=0.22)
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.3,
        help="Threshold for the optional fifth direct flag-segmentation channel",
    )
    parser.add_argument(
        "--flag-classifier-threshold",
        type=float,
        default=0.75,
        help="Minimum n816 probability when a flag curve is localized",
    )
    parser.add_argument(
        "--flag-fallback-threshold",
        type=float,
        default=0.9,
        help="Minimum n816 probability for a CNN-inferred flag without a localized curve",
    )
    parser.add_argument(
        "--disable-flag-classifier",
        action="store_true",
        help="Use geometry-only flag candidates without the pretrained n2/n4/n816 validator",
    )
    return parser.parse_args()


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def load_or_predict(
    image: np.ndarray,
    cache_path: Path,
    model_path: Path,
    model_kind: str,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        note_score = cached["note_score"].astype(np.float32)
        stem_score = cached["stem_score"].astype(np.float32)
        beam_score = cached["beam_score"].astype(np.float32) if "beam_score" in cached else None
        flag_score = cached["flag_score"].astype(np.float32) if "flag_score" in cached else None
        return note_score, stem_score, beam_score, flag_score

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix.lower() == ".onnx":
        probabilities = run_onnx_segmentation(model_path, image, batch_size=batch_size)
    elif model_path.suffix.lower() in {".keras", ".h5"}:
        probabilities = run_keras_segmentation(model_path, image, batch_size=batch_size)
    else:
        raise ValueError("Model must be .onnx, .keras, or .h5")

    if model_kind == "original":
        if probabilities.shape[-1] != 4:
            raise ValueError(
                f"Original 25-omr model must have four output channels, got {probabilities.shape[-1]}"
            )
        stem_score = probabilities[..., 1]
        note_score = probabilities[..., 2]
        beam_score = None
        flag_score = None
    else:
        if probabilities.shape[-1] not in {4, 5}:
            raise ValueError(
                "Primitive model must have four channels "
                "(background/notehead/stem/beam) or five with flag last; "
                f"got {probabilities.shape[-1]}"
            )
        note_score = probabilities[..., 1]
        stem_score = probabilities[..., 2]
        beam_score = probabilities[..., 3]
        flag_score = probabilities[..., 4] if probabilities.shape[-1] == 5 else None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note_score": note_score, "stem_score": stem_score}
    if beam_score is not None:
        payload["beam_score"] = beam_score
    if flag_score is not None:
        payload["flag_score"] = flag_score
    np.savez_compressed(cache_path, **payload)
    return note_score, stem_score, beam_score, flag_score


def process_image(
    image_path: Path,
    output_dir: Path,
    detector: PrimitiveDetector,
    args: argparse.Namespace,
) -> dict:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    processed, scale_x, scale_y = resize_for_model(
        original, detector.config.target_pixels
    )
    cache_path = output_dir / ".cache" / f"{image_path.stem}_segmentation.npz"
    if args.legacy_segmentation:
        if scale_x != 1.0 or scale_y != 1.0:
            raise ValueError("Legacy segmentation requires an image already matching the cache size")
        note_score, stem_score = load_legacy_segmentation(args.legacy_segmentation)
        beam_score = None
        flag_score = None
    else:
        note_score, stem_score, beam_score, flag_score = load_or_predict(
            processed,
            cache_path,
            args.model,
            args.model_kind,
            args.batch_size,
            args.force,
        )
    result = detector.detect(
        processed,
        note_score,
        stem_score,
        beam_score=beam_score,
        flag_score=flag_score,
        source=str(image_path.resolve()),
    )
    result = rescale_result(result, scale_x, scale_y)
    result["image"] = {"width": int(original.shape[1]), "height": int(original.shape[0])}
    result["model"] = {
        "path": str(args.model.resolve()) if not args.legacy_segmentation else None,
        "kind": args.model_kind if not args.legacy_segmentation else "legacy-25-omr-cache",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}.json"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    stem_overlay_path = output_dir / f"{image_path.stem}_stems.png"
    beam_overlay_path = output_dir / f"{image_path.stem}_beams.png"
    accidental_overlay_path = output_dir / f"{image_path.stem}_accidentals.png"
    result["outputs"] = {
        "json": str(json_path),
        "overlay": str(overlay_path),
        "stems": str(stem_overlay_path),
        "beams": str(beam_overlay_path),
        "accidentals": str(accidental_overlay_path),
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlay = draw_overlay(original, result)
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write {overlay_path}")
    stem_overlay = draw_stem_overlay(original, result)
    if not cv2.imwrite(str(stem_overlay_path), stem_overlay):
        raise RuntimeError(f"Failed to write {stem_overlay_path}")
    beam_overlay = draw_beam_overlay(original, result)
    if not cv2.imwrite(str(beam_overlay_path), beam_overlay):
        raise RuntimeError(f"Failed to write {beam_overlay_path}")
    accidental_overlay = draw_accidental_overlay(original, result)
    if not cv2.imwrite(str(accidental_overlay_path), accidental_overlay):
        raise RuntimeError(f"Failed to write {accidental_overlay_path}")
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for option_name, value in (
        ("--flag-classifier-threshold", args.flag_classifier_threshold),
        ("--flag-fallback-threshold", args.flag_fallback_threshold),
        ("--flag-threshold", args.flag_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{option_name} must be between 0 and 1")
    images = collect_images(args.input.expanduser(), args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print("No supported images found", file=sys.stderr)
        return 2
    if args.legacy_segmentation and len(images) != 1:
        print("--legacy-segmentation supports exactly one input image", file=sys.stderr)
        return 2

    config = DetectionConfig(
        notehead_threshold=args.notehead_threshold,
        stem_threshold=args.stem_threshold,
        flag_threshold=args.flag_threshold,
        flag_classifier_threshold=args.flag_classifier_threshold,
        flag_fallback_threshold=args.flag_fallback_threshold,
    )
    flag_classifier = None
    if not args.disable_flag_classifier:
        project_root = Path(__file__).resolve().parent
        flag_classifier = SingleStemRhythmClassifier(
            project_root / "25-omr/training/stemupImg32x32_best.pth",
            project_root / "25-omr/training/stemdownImg32x32_best.pth",
        )
    detector = PrimitiveDetector(config, flag_classifier=flag_classifier)
    failures = 0
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        try:
            result = process_image(image_path, args.output_dir, detector, args)
            counts = result["counts"]
            print(
                f"  noteheads={counts['noteheads']} stems={counts['stems']} beams={counts['beams']} "
                f"flags={counts.get('flags', 0)} accidentals={counts.get('accidentals', 0)} "
                f"staff_spacing={result['staff_spacing']}"
            )
        except Exception as exc:  # Keep a long batch running while reporting each failure.
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
