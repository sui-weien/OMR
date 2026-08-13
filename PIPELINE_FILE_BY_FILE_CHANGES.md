# File-by-file: what changed in each pipeline, in original step order

This document walks both upstream pipelines in their own documented step order
and states, file by file, whether this repository changed anything. Every
"unchanged" claim was verified with a real diff against the upstream project's
own git history, not assumed from memory — and every "adapted from" claim below
points at a specific function in a specific file, not a vague description.

**One note up front, for whoever reads this next**: two other local clones —
`OMR_layout` and `25-omr` — separately contain their own uncommitted piano-
adaptation edits, made directly on top of the upstream pipelines rather than
as a rewrite. None of that is merged into this repository; a couple of ideas
from it were deliberately re-implemented here instead (see "What we borrowed"
below), everything else in this repo is independent from-scratch code.

## Part A — `OMR_layout`'s pipeline (staff + instrumentation labeling)

`OMR_layout`'s own `run_pipeline.sh` defines the step order below. Every one of
these files, diffed directly against `OMR_layout`'s upstream git history, is
**byte-for-byte identical** to the original:

| Step | File | Produces | Changed? |
|---|---|---|---|
| 1 | `1_doctr_full_page_process.py` | `{stem}_ocr.json` (DocTR OCR of the left-margin instrument labels) | No — identical |
| 2a | `2_staff_box.py` | `{stem}.txt` (YOLO staff bounding boxes) | No — identical |
| 2b | `2_1_staff_box_plot.py` | `{stem}_box_plot.png` | No — identical |
| 2c | `2_cnt_staff_omr.py` | `{stem}_stafflist.pkl` (oemer-derived staff segmentation) | No — identical |
| 3a | `3_count_staff2json.py` | `{stem}_yolostaff.json` | No — identical |
| 3b | `3_check_staff_cnt_pk.py` | `{stem}_yolostaff_gt.json` | No — identical |
| 4 | `4_ocr_filter.py` | `{stem}_ocr_filtered.json` | No — identical |
| 5 | `5_ocr_filtered_plot.py` | `{stem}_ocr_filtered.png` | No — identical |
| 6 | `6_1_gpt_classify_reformat.py` | `{stem}_ocr_filtered_classified_normalized.json` | No — identical |
| 8 | `8_gpt_staffinfo.py` | `{stem}_staffgroup.json` (GPT Vision staff/instrument layout) | No — identical |
| b | `b_trans_gpt.py` | `{stem}_trans_gpt.csv` | No — identical |
| 7 | `7_matching_reformat_standard.py` | `{stem}_staff_instruments.json` + `_trans_gpt_modify.csv` | No — identical |
| f | `f.py` | `{stem}_trans_gpt_final.csv` | No — identical |
| — | `run_pipeline.sh` (orchestration) | runs the above in order | No — identical |
| — | `omr/__init__.py`, `omr/bbox.py`, `omr/layers.py`, `omr/logger.py`, `omr/staffline_extraction.py` | oemer-derived staff-segmentation helper package used by step 2c | No — every file identical |

We did not touch a single line of this pipeline. It answers "which instrument
plays on this staff" — a question a solo piano page doesn't have (always one
instrument, always two staves) — so we simply don't invoke it for a piano
score; we don't need to modify it to *not* need it.

*(Separately, and outside this repo: a local `OMR_layout` clone has real,
uncommitted edits to 5 of these files — `1_doctr_full_page_process.py`,
`2_staff_box.py`, `2_1_staff_box_plot.py`, `2_cnt_staff_omr.py`,
`3_check_staff_cnt_pk.py` — adding a music-notation-aware OCR tagger and a
"Grand_staff" YOLO class that pairs a treble+bass staff into right-hand/
left-hand tracks. None of that is reflected here; this repo's staff pairing
in Part B is a separate, from-scratch implementation of the same idea.)*

## Part B — `25-omr`'s pipeline (string-quartet segmentation → MusicXML)

A different situation: **none of `25-omr`'s own `.py` files exist in this
repository at all**, modified or otherwise. `25-omr/` is entirely git-ignored
here; the only two things ever pulled out of it are pretrained weight files —
the `seg_net` ONNX checkpoint and the single-stem rhythm CNN's `.pth` weights.
Every step of `25-omr`'s own pipeline was reimplemented from scratch in
`primitive_omr/`, because its logic assumes a hardcoded 4-track string-quartet
layout (see `STRING_QUARTET_TO_PIANO_ADAPTATION.md` for the evidence).

| `25-omr`'s original step | Original file/function | This repo's replacement | Relationship to the original |
|---|---|---|---|
| Rasterize PDF to page images | `pdf2png.py` | N/A | Not needed — this project takes JPG/PNG directly, no PDF input |
| Run segmentation CNNs, cache result | `omr/part1.py::runModel1` | `primitive_omr/inference.py` | New code. Calls the *same* `seg_net` ONNX checkpoint, with our own tiled-inference implementation |
| Staff/notehead/symbol extraction, cache `_staffList.pkl` | `omr/part2.py` | `primitive_omr/detector.py::detect_noteheads` + `primitive_omr/musicxml.py::detect_staffs` | New code. No `_staffList.pkl` cache — staff lines are re-derived from the image every run |
| Beam image + notehead/stem search | `pdf2musicXML.py::getBeamImage`, `getInitialNoteheadBoxList`, `getStemList` | `primitive_omr/detector.py::detect_stems`, `detect_beams` | New code. Contour shape analysis + Hough-line stem detection with note-guided recovery, instead of column-projection search + bar-height-scaled morphology |
| Rhythm classification per stem | `pdf2musicXML.py::knnRhythmAndDraw`, `get_prediction_singleStem.py::Single_Stem_Classifier` | `primitive_omr/rhythm_classifier.py::SingleStemRhythmClassifier` + `detector.py::detect_flags` | Reuses the *trained weights* unchanged; the calling convention, crop logic, and (as of this session's flag-stability fix) the requirement that real ink touch the stem tip before trusting the classifier, are all new |
| Rest detection | `Rest_Classifier` / `findRests` | Not implemented | Documented limitation |
| Clef/accidental classification | `get_prediction.py::Sfn_Clef_classifier` (7-class CNN incl. a literal `ViolaC` class) | `primitive_omr/detector.py::detect_accidentals`, `_classify_accidental_shape` | New code, new approach: a from-scratch geometry heuristic, not a trained classifier. (We also tested reusing `oemer`'s trained sfn SVM classifier directly — see `ACCIDENTAL_CLASSIFIER_EXPERIMENT.md` — it performed worse and was not adopted) |
| Pitch assignment | `pdf2musicXML.py::assignPitch` | `primitive_omr/musicxml.py::staff_y_to_pitch` | New code, and — unlike either upstream version — actually validated: 93.9% notehead recall against real ground truth, with real per-note pitch values, not a placeholder |
| **Barlines / measures** | `pdf2musicXML.py::findBarlines`, `constructBar` | `primitive_omr/musicxml.py::detect_barlines`, used inside `write_musicxml` | **Adapted from a real idea** (see "What we borrowed" below) |
| Dots/ties/tuplets | `assignDots`, dedicated tie/tuplet handling | Not implemented | Documented limitation |
| Export to MusicXML | `pdf2musicXML.py::exportXML` | `primitive_omr/musicxml.py::write_musicxml` (via `music21`) | New code. Alternating treble/bass 2-track grand staff instead of a hardcoded 4-track violin/violin/viola/cello structure |
| Entry point / orchestration | `pdf2musicXML.py`'s `__main__`, driven by a hand-written per-piece JSON (`tsChange`, `clef_options`, `numTrack`) | `detect_primitives.py` (primitives only), `image_to_musicxml.py` (full pipeline) | New code. No per-piece human JSON — everything is inferred from the image |
| Evaluation against ground truth | *(none in either upstream project)* | `evaluate_primitives.py` | Entirely new capability |
| Model fine-tuning | *(none)* | `fine_tune_primitives.py` | Entirely new capability |
| OCR-assisted dynamic/expression markings | *(none in `25-omr`; a broader, unvalidated version exists only in the unmerged `OMR_layout` clone)* | `primitive_omr/text_markings.py`, `detect_text_markings.py` | See "What we borrowed" below |
| Tests | *(none)* | `tests/test_primitive_detector.py`, `tests/test_musicxml_conversion.py`, `tests/test_text_markings.py` | Entirely new |

### What we borrowed (and what we deliberately didn't)

Two pieces of this repo's code did start from an idea found in those unmerged
clones, rather than being invented from scratch — worth calling out
explicitly since "0 files reused" is the norm everywhere else in Part B:

- **Barline detection**, in `primitive_omr/musicxml.py::detect_barlines`:
  a barline candidate is only trusted if the right-hand and left-hand staff
  of a system *independently* agree on the same x-position, which rejects
  stems/noteheads/other vertical ink that only crosses one staff. Reimplemented
  against this repo's own `StaffGeometry`/detector conventions rather than
  copied line-for-line, and barlines are kept strictly per-system (the
  original flattens all systems into one list and merges across them, which
  conflates unrelated systems). Validated: 298/299 systems (99.7%) got a
  real RH/LH-agreed barline grid across the 49-page ground truth set, with
  visual spot-checks across 3 pages confirming the detected positions line
  up with the actual printed barlines.

- **OCR-assisted text markings**, in `primitive_omr/text_markings.py`: OCR
  words are tagged as dynamics/expression markings by a small vocabulary
  rule set. Reimplemented and cut down after validating against real
  ground truth (only dynamic and expression markings survived; tempo,
  pedal, and fingering were dropped — see that file's docstring), and the
  vocabulary was expanded against real OCR output it was originally missing
  common words from (`sf`, `sempre`, `poco`, `mezza`, `voce`, `d'un`, `piu`, ...).

## Part C — Everything else new

`evaluate_primitives.py`, `fine_tune_primitives.py`, and the accidental
detector in `primitive_omr/detector.py` have no equivalent worth mapping to
either upstream pipeline at all — they exist because this project needed them,
not because something upstream was being replaced. All of today's validated
numbers (flag false-positive fix, beam/notehead/accidental recall, barline
coverage) came from `evaluate_primitives.py` against the same 49-page
ground-truth set referenced throughout this document
(`project-10-at-2026-07-23-01-03-bdef76e6` — a YOLO-labelled set of Beethoven
piano sonata pages, distinct from the `Xia` set the original project report
was written against).

## How to check any of this yourself

- `tests/` runs without needing any model checkpoints or ground-truth data.
- `evaluate_primitives.py <dataset> <predictions>` reproduces every recall/
  accuracy number cited in this document and in `STRING_QUARTET_TO_PIANO_ADAPTATION.md`,
  given the same ground-truth dataset.
- Every "unchanged" claim in Part A, and the "0 files present" claim in Part B,
  can be re-verified with `git diff HEAD -- <file>` inside a fresh clone of
  the relevant upstream repository — not against the local clones mentioned
  in the caveat above, which have their own uncommitted edits layered on top.
