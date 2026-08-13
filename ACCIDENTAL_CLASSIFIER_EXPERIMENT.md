# Tried: reusing oemer's accidental classifier — result was negative

`primitive_omr/detector.py`'s `_classify_accidental_shape()` guesses sharp/flat/natural
from hand-coded geometric rules (a two-vertical-stroke grid for sharp/natural, an
enclosed hole for flat's bowl). Its known weakness: when a symbol's ink is fragmented
by adaptive thresholding into several small disconnected pieces, those geometric
measurements degrade. This note records an experiment to check whether
[BreezeWhite/oemer](https://github.com/BreezeWhite/oemer) — the project whose
segmentation checkpoints `25-omr` (and, through it, this repo) already reuses — has a
better accidental classifier we could substitute in, and why the answer turned out to
be no, at least not as a drop-in.

## What oemer actually does

Investigated by reading oemer's committed source directly (not assumed from its README):

- Its CNN segmentation model does **not** distinguish sharp/flat/natural. All three
  (plus clefs) collapse into one shared binary channel, `clefs_keys`
  (`oemer/utils.py::generate_pred()`: `clefs_keys = np.where(sep==3, 1, 0)`).
- The actual sharp/flat/natural decision is made by a separate, classical **scikit-learn
  SVM** (`sklearn.svm.SVC`), trained on 40×70-pixel crops flattened to raw pixel vectors
  (`oemer/classifier.py::train()` / `predict()`), not a CNN and not geometric rules.
  The trained weights ship in the repo as a pickle: `oemer/sklearn_models/sfn.model`
  (8.68 MB), trained on crops from the DeepScores-extended dataset. Repo is MIT
  licensed, so reuse is permitted.
- Critically, that SVM is still classifying a **binarized** crop — sourced from oemer's
  own CNN segmentation mask, then eroded/dilated, not from adaptive thresholding like
  ours. So it doesn't sidestep the fragmentation risk in principle; it just sources its
  binary mask differently.

## The experiment

Loaded `sfn.model` directly (`pickle.load`, works under scikit-learn 1.9.0 despite being
trained on 1.2.0 — version-mismatch warning only, no failure) and ran its `predict()`
logic on the *same* candidate crops our own pipeline already extracts and matches
against real ground truth (project-10, 49 annotated Beethoven piano pages), so both
classifiers were scored on identical inputs and identical matched instances:

| Classifier | Type accuracy (matched instances only) | Confusion pattern |
|---|---:|---|
| Our geometric heuristic | 68.75% (11/16) | Uncertain cases fall to `unknown` (applies no pitch alteration — a safe failure) |
| oemer's SVM | 50.0% (8/16) | Systematically biased toward "natural": 6/9 sharps and 2/3 flats misclassified as natural |

## Why it performed worse, not better

The SVM is a shallow, pixel-vector classifier with no built-in invariance to
translation/scale/binarization-style differences — unlike a CNN, it has no learned
feature hierarchy to fall back on when the input distribution shifts. Feeding it crops
from *our* candidate-detection logic (our own bbox conventions, margins, and adaptive-
threshold binarization) rather than the oemer-pipeline-specific crops it was actually
trained on amounts to a real train/inference distribution shift, and the systematic
"natural" bias suggests the model is failing in a structured, not random, way under
that shift.

## Conclusion

Not adopted. Reusing oemer's `sfn.model` as a drop-in replacement made accidental type
accuracy worse (50% vs. our existing 68.75%), not better. Making it usable would require
either (a) re-cropping to exactly match oemer's own segmentation-mask-derived crop
convention before classification, or (b) retraining the SVM (or a small CNN) directly on
crops produced by *this* project's own candidate-detection pipeline. Both are
meaningfully larger undertakings than "swap in a pretrained classifier," so this path is
parked rather than pursued further for now.
