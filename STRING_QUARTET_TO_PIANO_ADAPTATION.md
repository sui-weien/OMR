# From String-Quartet OMR to Piano OMR: What Actually Changed

This document explains, with references to the actual upstream source code, what had
to change to turn [lattellie/25-omr](https://github.com/lattellie/25-omr) (built for
string quartets) and [Bobo1111111/OMR_layout](https://github.com/Bobo1111111/OMR_layout)
(built for full orchestral scores) into the solo-piano pipeline in this repository
(`primitive_omr/`, `detect_primitives.py`, `image_to_musicxml.py`).

The short version: we did not patch either upstream pipeline's own note/stem/beam/pitch
logic. We kept exactly one asset from `25-omr` — the pretrained `seg_net` ONNX
segmentation model's raw probability output — and threw away every piece of code built
around it, because that code encodes assumptions specific to string-quartet engraving
that do not hold for piano scores. We kept nothing executable from `OMR_layout` except
its general idea (YOLO + OCR + GPT to label staff instrumentation), because piano scores
don't have an "instrumentation" question to answer in the first place.

## 1. What `25-omr` actually is (verified against its committed source)

`25-omr`'s root script, `pdf2musicXML.py`, is a ~3700-line, single-purpose pipeline for
**string quartets specifically**, not a generic OMR engine with a quartet demo bolted on.
The instrumentation assumption is load-bearing, not incidental:

```python
# 25-omr/pdf2musicXML.py (original)
NUM_TRACK = 4
TRACK_SHIFT = [0, 0, 0, 0]
CLEF_OPTIONS = [[1], [1], [0, 1], [-1, -2, 1]]
```

Its own README documents the clef codes by instrument name: `1: treble, 0: alto (viola),
-1: bass, -2: tenor (cello)`, and the per-piece config schema hardcodes `"numTrack": 4`
with clef options like `[[1],[1],[0],[-1,-2]]` — violin I, violin II, viola, cello, in
that order, every time. `omr/part4.py` has a clef-code comment written in exactly those
terms: `# clef: c1 for gclef(vln), c2 for fclef(cello)`. The accidental/clef classifier
(`get_prediction.py::Sfn_Clef_classifier`) has `class_names = ['BassF', 'ViolaC', 'flat',
'natural', 'noClass', 'sharp', 'trebleG']` — the alto clef is a training class literally
named `ViolaC`. There is no "2 staves, alternating treble/bass, grand staff" concept
anywhere in the code; the entire pipeline structurally expects four independent
single-line instrumental parts, not one instrument reading two staves simultaneously.

Per-piece human input is also part of the design, not an artifact of an incomplete demo:
every score needs a hand-written JSON (`jsonTemplate.json`) specifying `numPage`,
`tsChange` (page/measure locations where the time signature changes), and the
`clef_options` above. Two Beethoven quartet movements bundled with the repo
(`string_dataset/pdf_data/beethoven1/beethoven1.json`) are the only pieces that actually
run out of the box.

`pdf2musicXML.py` itself contains extensive bespoke geometry code layered on top of the
segmentation model, all tuned for this repertoire's typical spacing and density:
`getBeamImage()` erodes/dilates with a *wide horizontal* kernel sized to the estimated
bar height, an approach that assumes beams are the dominant horizontal-adjacent-to-stem
feature and stems are comparatively sparse per staff. `getInitialNoteheadBoxList()` /
`getStemList()` find a stem's x-position by scanning column-sum projections in a search
window to the immediate left/right of each notehead — reasonable for single-voice string
parts, brittle for dense two-hand piano chords where several stems and noteheads sit
within a couple of note-widths of each other. `knnRhythmAndDraw()` classifies each stem
crop with a tiny pretrained CNN (`Single_Stem_Classifier`, 3 classes: `n2`/`n4`/`n816`)
whose own demo code loads training crops named `bach_*.jpg` — evidence for what
repertoire and engraving style it actually learned from, and it isn't piano.

The one piece of `25-omr` this repository actually keeps is the **raw segmentation
model**, called through the shared `omr.inference.inference()` wrapper:

```python
# 25-omr/omr/part2.py (original) — what the four seg_net channels mean
sep, _ = inference(os.path.join(MODULE_PATH, "checkpoints/seg_net"), img_path, ...)
stems_rests = np.where(sep == 1, 1, 0)
notehead    = np.where(sep == 2, 1, 0)
clefs_keys  = np.where(sep == 3, 1, 0)
```

`primitive_omr/inference.py` calls the same ONNX checkpoint directly (no tiling/staff
logic borrowed from `25-omr` — that part is our own code, see §3), and
`primitive_omr/rhythm_classifier.py` reuses the *weights* of the same single-stem CNN,
but not `knnRhythmAndDraw`'s grouping logic around it. Everything else — staff
detection, notehead/stem/beam extraction, pitch assignment, accidental detection,
MusicXML export — is new code written for this project.

## 2. What `OMR_layout` actually is

`OMR_layout`'s pipeline (the numbered scripts `1_doctr_full_page_process.py` through
`f.py` in this repo, kept close to their original form) answers a different question
than `25-omr`: not "what pitch and rhythm is this note," but "which instrument is
playing on this staff." It was built for full orchestral scores where a system can have
a dozen or more staves (flutes, oboes, horns, trumpets, timpani, strings...), and the
useful output is a CSV mapping each staff to an instrument name, part number, and
transposition — not a MusicXML performance transcription. Its pipeline is DocTR OCR to
read the instrument labels printed at the left margin, YOLO to detect staff bounding
boxes, GPT to normalize the OCR text ("Corni in Es." → instrument `Horn`, tone `E flat`)
and to infer the system/staff-group layout from the page image directly, then a set of
matching heuristics to attach the right label to the right staff.

None of that question exists for solo piano: there is exactly one instrument, and it
always occupies exactly two staves per system (treble above, bass below). There is no
instrument name printed anywhere to OCR, no GPT normalization needed, no per-staff
instrumentation CSV to produce. We did not adapt `OMR_layout`'s code for piano; we simply
don't run that half of the combined pipeline when the input is a piano score. What this
repository actually reuses from `OMR_layout` conceptually — not as ported code — is the
idea of using a YOLO-detected staff layer to inform region-of-interest cropping;
`detect_staffs()` in `primitive_omr/musicxml.py` re-derives staff line positions from
scratch, directly from long-horizontal-stroke projections in the page image, because a
solo piano page doesn't need OCR-driven staff grouping at all.

## 3. What `primitive_omr/` actually builds, and why the old code couldn't be patched

| Need | `25-omr`'s approach (string quartet) | This project's approach (piano) | Why the old one doesn't transfer |
|---|---|---|---|
| Track/clef structure | Hardcoded `NUM_TRACK=4`, `CLEF_OPTIONS` per named instrument (violin/violin/viola/cello) | Generic alternating treble/bass 2-track grand staff, inferred per detected staff, no instrument names involved | Piano has one performer reading two staves, not four independent single-line parts; there's no "which instrument" to hardcode |
| Per-piece config | Hand-written JSON: page count, time-signature-change locations, clef options | None — everything inferred from the image alone | The whole point of `25-omr`'s JSON was encoding facts a human already knows about a specific quartet score; that doesn't scale to "run this on any piano page" |
| Staff/cache state | `_staffList.pkl`, computed once by `omr/part2.py`, then reused as a cache across reruns | Recomputed from the page image every run in `detect_staffs()` | We don't reuse `part2.py`'s staff/symbol extraction at all, so there's nothing to cache from it |
| Notehead/stem detection | Column-projection search around each notehead (`getInitialNoteheadBoxList`/`getStemList`), tuned for single-voice string spacing | Thresholded segmentation mask → connected components → Hough-line stem detection with note-guided recovery, tuned for dense chorded piano writing | Piano chords put several stems and noteheads within a note-width or two of each other; a projection search calibrated for one voice per staff misreads that density |
| Beam detection | Wide horizontal morphological erosion/dilation sized to bar height | Contour shape analysis (aspect ratio, fill ratio) plus a stem-endpoint bridging pass for partial/single-note beam fragments | Different failure mode entirely — piano beam runs are shorter and more irregular (broken chords, isolated subdivided notes) than typical string-quartet beaming |
| Accidental detection | A trained clef+accidental classifier (`Sfn_Clef_classifier`) with 7 fixed classes including a literal `ViolaC` class, tuned to quartet clef layout | A from-scratch geometry heuristic (`detect_accidentals` in `primitive_omr/detector.py`): search a window left of each notehead, classify sharp/natural by a two-vertical-stroke grid shape, flat by an enclosed-hole bowl shape | `25-omr` never needed piano-generic accidental detection decoupled from a fixed clef list; this didn't exist for piano at all until this session added it |
| Rhythm/flag classification | `knnRhythmAndDraw`'s grouping logic feeding the single-stem CNN | Our own flag geometry pipeline, using the *same pretrained CNN weights* only as an optional secondary check, gated behind a requirement that real ink actually touches the stem tip | The CNN was trained on isolated quartet-style note+stem crops; trusting its confidence alone (without requiring ink evidence) is exactly the bug this session's flag-stability fix found and closed |

## 4. Where today's findings connect back to "trained for string quartets"

This isn't just an architectural argument — this session found a concrete, measurable
symptom of it. Evaluating notehead/stem detection against real piano ground truth
(49 annotated Beethoven piano sonata pages), two of four remaining stem misses traced to
notes whose *ink was large enough* to clear our detector's own size filters, but whose
segmentation-model confidence never crossed the acceptance threshold in the first place
— i.e., the `seg_net` model itself didn't recognize the ink as "notehead" with enough
confidence, for compact note groupings that are common in piano writing but may be
under-represented in whatever repertoire actually trained `seg_net` (a quartet-focused
oemer-derived model, per §1). That's a model-capability gap, not a threshold we can tune
away in `primitive_omr`'s post-processing — the same category of limitation that
`Sfn_Clef_classifier`'s `ViolaC` class name makes obvious for accidentals: these upstream
components were built and trained for a specific four-part string layout, and piano
notation exercises them outside that training distribution.

A properly-scoped fix (discussed but not started this session, given the risk of
naively fine-tuning on sparse annotations — see the project's fine-tuning safety check
in `fine_tune_primitives.py`) would fine-tune `seg_net`'s head using real piano ground
truth, with a loss that doesn't penalize the model for unlabeled-but-real noteheads
elsewhere on the page.

## 5. Summary

`25-omr` and `OMR_layout` supplied two things we kept: a pretrained ONNX segmentation
model (notehead/stem-rest/clef-key channels) and a pretrained single-stem rhythm CNN.
Everything that turns those raw signals into notes, chords, beams, accidentals, and a
piano-appropriate two-staff MusicXML score — `primitive_omr/detector.py`,
`primitive_omr/musicxml.py`, `primitive_omr/inference.py`,
`primitive_omr/rhythm_classifier.py`'s calling convention — is new code, because the
existing string-quartet and orchestral-layout code encoded structural assumptions
(four fixed instrumental parts with named clefs, per-piece human JSON, OCR-based
instrument labeling) that don't describe what a piano page looks like.
