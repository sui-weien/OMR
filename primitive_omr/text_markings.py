"""OCR-assisted detection of dynamic and expression text markings.

This is a conservative, experimental addition: OCR each page for printed
text ("cresc.", "dolce", "sf", ...) and tag words that clearly fall in one
of two categories -- `dynamic_mark` (p, f, cresc., sf, ...) and
`expression_mark` (dolce, sempre, poco, ...). Every other category from the
original rule set this was adapted from (tempo markings, pedal markings,
fingering numbers) is deliberately NOT exposed here, because validating
against real ground truth (project-10, 49 annotated piano pages) showed
they are not reliable enough to ship:

- Pedal markings ("Ped.") are misread by OCR into unrelated garbage text
  on this font/engraving style (confirmed 0% correct out of 33 located
  instances) -- no amount of vocabulary tuning fixes a wrong OCR read.
- Fingering numbers are tiny superscript-style digits that general-purpose
  document OCR mostly fails to even locate (only ~21% recall out of 2918
  annotated instances).

Dynamic and expression markings, by contrast, validated reasonably well:
OCR located 100% of annotated expression terms and 32% of annotated
dynamic marks, and after tuning the vocabulary against real OCR output,
type classification accuracy on located instances was 74.5% (expression)
and 72.1% (dynamic). Still imperfect -- verify before use -- but usable.

This module requires `python-doctr`, which is not a dependency of the main
detection pipeline (it needs a newer torch than `requirements_piano_omr.txt`
pins) and is only ever imported lazily, inside `detect_text_markings`, so
importing this module elsewhere does not require doctr to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DYNAMIC_MARKS = {
    "p", "f", "mf", "mp", "ff", "pp", "fff", "ppp",
    "sfz", "sf", "fp", "rfz", "cresc.", "dim.", "decresc.",
}
EXPRESSION_MARKS = {
    "dolce", "cantabile", "espressivo", "legato", "staccato", "sostenuto",
    "grazioso", "con", "brio", "moto", "in", "poco", "mezza", "voce",
    "sempre", "d'un", "piu", "più", "tenute", "subito", "attacca",
    "manontroppo",
}
# Page headers (title/composer) sit near the top of the page and would
# otherwise collide with the single-word marks above (e.g. a composer
# initial or a word like "Sonata" is not a dynamic or expression marking).
_HEADER_Y_FRACTION = 0.08


def classify_text_marking(text: str, y_min: float) -> str | None:
    """Return "dynamic_mark", "expression_mark", or None.

    `y_min` is the token's top edge as a fraction of page height (0-1),
    matching DocTR's normalized word geometry.
    """
    if y_min < _HEADER_Y_FRACTION:
        return None
    cleaned = text.strip().lower()
    if not cleaned:
        return None
    if cleaned in DYNAMIC_MARKS:
        return "dynamic_mark"
    if cleaned in EXPRESSION_MARKS:
        return "expression_mark"
    return None


def detect_text_markings(image_path: str | Path, model: Any | None = None) -> dict[str, Any]:
    """Run OCR on one page image and return only confident dynamic/expression tags.

    Requires `python-doctr` to be installed in the current environment
    (see requirements_ocrt.txt) -- imported lazily so this module can be
    imported without it. Pass an already-built `ocr_predictor()` as `model`
    to reuse it across a batch instead of reloading weights per call.
    """
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError as exc:
        raise RuntimeError(
            "python-doctr is required for OCR-based text-marking detection; "
            "install requirements_ocrt.txt in a dedicated environment."
        ) from exc

    if model is None:
        model = ocr_predictor(pretrained=True)

    image_path = Path(image_path)
    doc = DocumentFile.from_images(str(image_path))
    result = model(doc)

    markings: list[dict[str, Any]] = []
    ocr_token_count = 0
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    ocr_token_count += 1
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    category = classify_text_marking(word.value, float(y_min))
                    if category is None:
                        continue
                    markings.append(
                        {
                            "text": word.value,
                            "category": category,
                            "confidence": round(float(word.confidence), 4),
                            "bbox_minmax": [
                                float(x_min),
                                float(y_min),
                                float(x_max),
                                float(y_max),
                            ],
                        }
                    )

    return {
        "filename": image_path.stem,
        "ocr_token_count": ocr_token_count,
        "markings": markings,
        "counts": {
            "dynamic_mark": sum(1 for m in markings if m["category"] == "dynamic_mark"),
            "expression_mark": sum(1 for m in markings if m["category"] == "expression_mark"),
        },
    }
