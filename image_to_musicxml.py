#!/usr/bin/env python3
"""Complete image -> primitive overlay -> MusicXML pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from detect_primitives import collect_images, process_image
from primitive_omr.detector import DetectionConfig, PrimitiveDetector
from primitive_omr.musicxml import convert_detection_to_musicxml
from primitive_omr.rhythm_classifier import SingleStemRhythmClassifier


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detect notehead/stem/beam, create an overlay, then export MusicXML."
    )
    parser.add_argument("input", type=Path, help="Input score image or directory")
    parser.add_argument("--output-dir", type=Path, default=root / "musicxml_output")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "25-omr/omr/checkpoints/seg_net/model.onnx",
    )
    parser.add_argument("--model-kind", choices=("original", "primitive"), default="original")
    parser.add_argument("--staff-mode", choices=("piano", "treble", "bass"), default="piano")
    parser.add_argument("--time-signature", default="4/4", help="Fallback, for example 4/4 or 3/4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Ignore cached segmentation")
    parser.add_argument("--notehead-threshold", type=float, default=0.28)
    parser.add_argument("--stem-threshold", type=float, default=0.22)
    parser.add_argument("--flag-threshold", type=float, default=0.30)
    parser.add_argument("--flag-classifier-threshold", type=float, default=0.75)
    parser.add_argument("--flag-fallback-threshold", type=float, default=0.90)
    parser.add_argument("--disable-flag-classifier", action="store_true")
    return parser.parse_args()


def _time_signature(value: str) -> tuple[int, int]:
    try:
        beats_text, beat_type_text = value.split("/", 1)
        beats, beat_type = int(beats_text), int(beat_type_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("--time-signature must look like 4/4 or 3/8") from exc
    if beats < 1 or beat_type not in {1, 2, 4, 8, 16}:
        raise ValueError("Unsupported --time-signature")
    return beats, beat_type


def main() -> int:
    args = parse_args()
    beats, beat_type = _time_signature(args.time_signature)
    images = collect_images(args.input.expanduser(), args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print("No supported images found", file=sys.stderr)
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
        root = Path(__file__).resolve().parent
        flag_classifier = SingleStemRhythmClassifier(
            root / "25-omr/training/stemupImg32x32_best.pth",
            root / "25-omr/training/stemdownImg32x32_best.pth",
        )
    detector = PrimitiveDetector(config, flag_classifier=flag_classifier)
    args.legacy_segmentation = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}: primitives -> overlay -> MusicXML", flush=True)
        try:
            detection = process_image(image_path, args.output_dir, detector, args)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not read {image_path}")
            primitive_overlay = cv2.imread(detection["outputs"]["overlay"], cv2.IMREAD_COLOR)
            if primitive_overlay is None:
                raise ValueError("Primitive overlay was not created")
            xml_path = args.output_dir / f"{image_path.stem}.musicxml"
            xml_overlay_path = args.output_dir / f"{image_path.stem}_musicxml_overlay.png"
            summary_path = args.output_dir / f"{image_path.stem}_conversion.json"
            summary = convert_detection_to_musicxml(
                image,
                detection,
                xml_path,
                xml_overlay_path,
                primitive_overlay,
                staff_mode=args.staff_mode,
                beats=beats,
                beat_type=beat_type,
            )
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            detection["outputs"].update(
                {
                    "musicxml": str(xml_path),
                    "musicxml_overlay": str(xml_overlay_path),
                    "conversion": str(summary_path),
                }
            )
            Path(detection["outputs"]["json"]).write_text(
                json.dumps(detection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"  staffs={summary['counts']['staffs']} events={summary['counts']['events']} "
                f"pitches={summary['counts']['pitches']}"
            )
            print(f"  overlay: {xml_overlay_path}")
            print(f"  XML:     {xml_path}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
