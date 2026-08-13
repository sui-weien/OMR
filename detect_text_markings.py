# env: ocrt
"""OCR-assisted dynamic/expression text-marking detection for one page or a folder.

Conservative and experimental: only reports "dynamic_mark" (p, f, cresc.,
sf, ...) and "expression_mark" (dolce, sempre, poco, ...) tags, each
validated against real ground truth -- see primitive_omr/text_markings.py
for why pedal markings and fingering numbers are deliberately not
included. Requires python-doctr (requirements_ocrt.txt), which is not a
dependency of the main detect_primitives.py / image_to_musicxml.py
pipeline, so this is meant to be run in its own environment as a separate,
optional step -- its output is not yet merged into the MusicXML export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from primitive_omr.text_markings import detect_text_markings

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


def process_image(model, image_path: Path, output_dir: Path) -> None:
    result = detect_text_markings(image_path, model=model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_text_markings.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = result["counts"]
    print(
        f"  ocr_tokens={result['ocr_token_count']} "
        f"dynamic_mark={counts['dynamic_mark']} expression_mark={counts['expression_mark']}"
    )
    print(f"  saved: {output_path}")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect OCR-derived dynamic/expression text markings in one image or a folder."
    )
    parser.add_argument("input", type=Path, help="Image file or directory of images")
    parser.add_argument("--output-dir", type=Path, default=Path("text_marking_output"))
    args = parser.parse_args()

    images = collect_images(args.input.expanduser())
    if not images:
        print("No supported images found")
        return 2

    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True)
    failures = 0
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        try:
            process_image(model, image_path, args.output_dir)
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
