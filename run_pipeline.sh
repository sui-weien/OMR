#!/usr/bin/env bash
# ============================================================
# OMR Pipeline — run all steps for a single image folder.
#
# Usage:
#   bash run_pipeline.sh <input_folder>
#
# Each step writes its output into <input_folder>.
# ============================================================
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

FOLDER="${1%/}"   # strip trailing slash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment helpers ──────────────────────────────────────
# conda activate does NOT work inside non-interactive scripts.
# Use "conda run" instead — it launches each command in the right
# env without needing to source conda.sh or activate anything.
#
# If all packages are merged into one env, set its name here:
ENV_NAME="omr_all"   # ← change to your merged env name
PYTHON_OCR="conda run --no-capture-output -n $ENV_NAME python"
PYTHON_STAVE="conda run --no-capture-output -n $ENV_NAME python"
PYTHON_INTERN="conda run --no-capture-output -n $ENV_NAME python"
PYTHON_OMR="conda run --no-capture-output -n $ENV_NAME python"
#
# If you use separate envs, set each name individually:
# PYTHON_OCR="conda run --no-capture-output -n ocrt python"
# PYTHON_STAVE="conda run --no-capture-output -n stave python"
# PYTHON_INTERN="conda run --no-capture-output -n intern python"
# PYTHON_OMR="conda run --no-capture-output -n omr10 python"

step() {
    echo ""
    echo "════════════════════════════════════════"
    echo "  STEP $1 — $2"
    echo "════════════════════════════════════════"
}

echo "Pipeline folder: $FOLDER"

# ── 1. DocTR OCR  [env: ocrt] ────────────────────────────────
# Output: {stem}_ocr.json
step "1" "DocTR OCR  [env: ocrt]"
$PYTHON_OCR "$DIR/1_doctr_full_page_process.py" "$FOLDER"

# ── 2a. YOLO staff detection  [env: stave] ───────────────────
# Output: {stem}.txt  (YOLO bounding boxes)
step "2a" "YOLO staff detection  [env: stave]"
$PYTHON_STAVE "$DIR/2_staff_box.py" "$FOLDER"

# ── 2b. YOLO result visualisation  [env: stave] ──────────────
# Output: {stem}_box_plot.png
step "2b" "YOLO box plot  [env: stave]"
$PYTHON_STAVE "$DIR/2_1_staff_box_plot.py" "$FOLDER"

# ── 2c. OMR staff segmentation  [env: omr10] ─────────────────
# Output: {stem}_stafflist.pkl
step "2c" "OMR staff segmentation  [env: omr10]"
$PYTHON_OMR "$DIR/2_cnt_staff_omr.py" "$FOLDER"

# ── 3a. YOLO txt → yolostaff.json  [env: stave] ──────────────
# Output: {stem}_yolostaff.json
step "3a" "YOLO txt → yolostaff.json  [env: stave]"
$PYTHON_STAVE "$DIR/3_count_staff2json.py" "$FOLDER"

# ── 3b. PKL → yolostaff_gt.json  [env: stave] ────────────────
# Output: {stem}_yolostaff_gt.json
step "3b" "PKL → yolostaff_gt.json  [env: stave]"
$PYTHON_STAVE "$DIR/3_check_staff_cnt_pk.py" "$FOLDER"

# ── 4. OCR filter  [env: stave] ──────────────────────────────
# Output: {stem}_ocr_filtered.json
step "4" "OCR filter  [env: stave]"
$PYTHON_STAVE "$DIR/4_ocr_filter.py" "$FOLDER"

# ── 5. OCR filter plot  [env: stave] ─────────────────────────
# Output: {stem}_ocr_filtered.png
step "5" "OCR filter plot  [env: stave]"
$PYTHON_STAVE "$DIR/5_ocr_filtered_plot.py" "$FOLDER"

# ── 6. GPT token classification  [env: intern] ───────────────
# Output: {stem}_ocr_filtered_classified_normalized.json
step "6" "GPT token classification  [env: intern]"
$PYTHON_INTERN "$DIR/6_1_gpt_classify_reformat.py" "$FOLDER"

# ── 8. GPT Vision staff info  [env: intern] ──────────────────
# Output: {stem}_staffgroup.json
step "8" "GPT Vision staff info  [env: intern]"
$PYTHON_INTERN "$DIR/8_gpt_staffinfo.py" "$FOLDER"

# ── b. GPT staffgroup JSON → CSV  [env: stave] ───────────────
# Output: {stem}_trans_gpt.csv
step "b" "GPT staffgroup JSON → CSV  [env: stave]"
$PYTHON_STAVE "$DIR/b_trans_gpt.py" "$FOLDER"

# ── 7. Staff matching and reformat  [env: stave] ─────────────
# Output: {stem}_staff_instruments.json  +  {stem}_trans_gpt_modify.csv
step "7" "Staff matching and reformat  [env: stave]"
$PYTHON_STAVE "$DIR/7_matching_reformat_standard.py" "$FOLDER"

# ── f. Final CSV post-processing  [env: stave] ───────────────
# Output: {stem}_trans_gpt_final.csv
step "f" "Final CSV post-processing  [env: stave]"
$PYTHON_STAVE "$DIR/f.py" "$FOLDER"

echo ""
echo "════════════════════════════════════════"
echo "  All steps complete."
echo "════════════════════════════════════════"
