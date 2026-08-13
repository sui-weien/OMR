# Piano OMR: Image → Primitive Overlay → MusicXML

This repository adapts two upstream OMR projects into an image-only piano-score workflow:

- [Bobo1111111/OMR_layout](https://github.com/Bobo1111111/OMR_layout): full-page OCR, staff layout, and orchestral instrumentation pipeline.
- [lattellie/25-omr](https://github.com/lattellie/25-omr): segmentation models, rhythm classifiers, and MusicXML experience for string-quartet OMR.

Our added pipeline detects `notehead`, `stem`, and `beam`, builds their relationships, creates diagnostic overlays, assigns piano staves and natural pitches, and writes a parseable MusicXML draft.

## Current scope

```text
JPG / PNG
   ↓
25-omr ONNX segmentation
   ↓
notehead / stem / beam / experimental flag detection
   ↓
primitive JSON + diagnostic overlays
   ↓
piano staff / pitch / basic rhythm inference
   ↓
conversion JSON + MusicXML
```

Inference needs only the image. It does not read YOLO annotations, `stafflist.pkl`, or an OpenAI API key. Dataset annotations are used only for offline evaluation and future fine-tuning.

## Clone

```bash
git clone --recurse-submodules https://github.com/itsivyma/piano-omr.git
cd piano-omr
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Environment

The verified environment is macOS with Python 3.11:

```bash
conda create -n omr25-311 python=3.11 -y
conda activate omr25-311
python -m pip install --upgrade pip
python -m pip install -r requirements_piano_omr.txt
```

Verify imports:

```bash
python -c "import torch, cv2, onnxruntime, music21, scipy; print('environment ready')"
```

Required upstream model files are supplied by the `25-omr` submodule:

```text
25-omr/omr/checkpoints/seg_net/model.onnx
25-omr/training/stemupImg32x32_best.pth
25-omr/training/stemdownImg32x32_best.pth
```

## Run the complete pipeline

```bash
python image_to_musicxml.py \
  /path/to/score.jpg \
  --output-dir musicxml_output/score
```

For a known fallback time signature:

```bash
python image_to_musicxml.py score.jpg \
  --time-signature 3/4 \
  --output-dir musicxml_output/score
```

The default mode pairs alternating treble and bass staves as a two-part piano score. Single-staff modes are also available:

```bash
python image_to_musicxml.py score.jpg --staff-mode treble
python image_to_musicxml.py score.jpg --staff-mode bass
```

## Primitive-only diagnostics

```bash
python detect_primitives.py \
  /path/to/score.jpg \
  --output-dir primitive_output/score
```

## Outputs

```text
<name>.json
<name>_overlay.png
<name>_stems.png
<name>_beams.png
<name>_musicxml_overlay.png
<name>_conversion.json
<name>.musicxml
.cache/<name>_segmentation.npz
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Evaluation

```bash
python detect_primitives.py /path/to/Xia/images \
  --output-dir primitive_output/xia

python evaluate_primitives.py /path/to/Xia primitive_output/xia
```

On the sparsely annotated `Beethoven_Op101-01-07.jpeg` page, the current annotated-target recall is:

| Primitive | Annotated | Matched | Recall |
|---|---:|---:|---:|
| notehead | 58 | 58 | 100.0% |
| stem | 58 | 53 | 91.4% |
| beam | 12 | 11 | 91.7% |

These are sparse-label recall values, not full-page precision or F1 scores.

## Important limitations

The current MusicXML is a reviewable draft. The new conversion layer does not yet fully recognise:

- key signatures and accidentals;
- printed clef or time-signature changes;
- rests, dots, ties, tuplets, and multiple voices;
- exact barlines and measure boundaries;
- cross-staff notation.

The upstream ONNX and PyTorch model weights have not been retrained in this version. Current accuracy improvements primarily come from staff-spacing normalisation, notehead splitting, Hough-line deduplication, note-guided stem recovery, barline rejection, thick beam-bridge validation, multi-beam separation, and primitive relationship modelling.

## Documentation

- [Project report](PIANO_OMR_PROJECT_REPORT.md)
- [Primitive detection and evaluation](PRIMITIVE_DETECTION.md)
- [Image-to-MusicXML guide](IMAGE_TO_MUSICXML.md)

## Upstream acknowledgement

This work is derived from and integrates [OMR_layout](https://github.com/Bobo1111111/OMR_layout) and [25-omr](https://github.com/lattellie/25-omr). The `25-omr` project states that its segmentation models originate from [Oemer](https://github.com/BreezeWhite/oemer).

Before making this derived repository public or redistributing model weights, review the licences and redistribution terms of every upstream project and model asset.
