# 圖片自動辨識 notehead、stem、beam、flag

## 目標

輸入一張乾淨樂譜圖片，自動輸出：

- notehead bounding boxes
- stem line endpoints
- beam polygons
- flag bounding boxes 與符尾數量
- notehead → stem 關聯
- stem → beam 關聯
- stem → flag 關聯
- JSON 結果與可視化 PNG
- 接續產生帶 staff／音高的 overlay 與 MusicXML

`detect_primitives.py` 負責圖像基本元件；完整的圖片到 MusicXML 流程請執行 `image_to_musicxml.py`。詳細說明見 `IMAGE_TO_MUSICXML.md`。

## 一個指令產生 overlay 與 MusicXML

```bash
python image_to_musicxml.py \
  /path/to/score.jpeg \
  --output-dir musicxml_output
```

輸出包含 primitive overlay、staff／音高 overlay、結構化 JSON 與 `.musicxml`。這條流程推論時只需要圖片，不使用訓練標註。

## 單張圖片推論

啟用環境：

```bash
conda activate omr25-311
cd /Users/shih/Desktop/OMR_layout
```

使用原始 `25-omr` segmentation checkpoint：

```bash
python detect_primitives.py \
  /path/to/score.jpeg \
  --output-dir primitive_output
```

第一次會執行 ONNX 模型；之後會沿用輸出目錄中的 `.cache/*_segmentation.npz`。

## 批次處理 Xia 圖片

```bash
python detect_primitives.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images' \
  --output-dir primitive_output/xia
```

先測一張：

```bash
python detect_primitives.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images' \
  --output-dir primitive_output/xia \
  --limit 1
```

Apple Silicon 目前預設使用 CPU ONNX provider。49 張圖片全部執行會花較長時間，建議先以 `--limit` 驗證。

## 輸出格式

每張輸入會產生：

```text
primitive_output/
├── <image>.json
├── <image>_overlay.png
├── <image>_stems.png
├── <image>_beams.png
└── .cache/
    └── <image>_segmentation.npz
```

JSON 主要欄位：

```json
{
  "noteheads": [
    {"id": "notehead-0", "bbox": [0, 0, 10, 10], "center": [5, 5], "confidence": 0.9}
  ],
  "stems": [
    {"id": "stem-0", "line": [10, 0, 10, 30], "confidence": 0.8}
  ],
  "beams": [
    {"id": "beam-0", "polygon": [[0, 0], [20, 0], [20, 5], [0, 5]]}
  ],
  "relations": {
    "notehead_to_stem": [{"notehead_id": "notehead-0", "stem_id": "stem-0"}],
    "stem_to_beam": [{"stem_id": "stem-0", "beam_id": "beam-0"}]
  }
}
```

`*_stems.png` 是 stem 專用診斷圖：綠線代表模型與幾何一致，紅線代表只由幾何補回，黃色框代表尚未連到 stem 的 notehead。

`*_beams.png` 是 beam／flag 專用診斷圖：綠框代表實心形狀與 stem 端點都支持，藍框代表由多個 stem 共同補回；紫框是有定位到曲線的 flag，橘框是高信心 CNN 推定的 flag。

## Notehead、stem、beam 準確率改善

目前的三層關係式流程：

1. `notehead`：使用原始 segmentation 機率圖、連通元件拆分與 IoU 去重。
2. `stem`：合併同一根粗 stem 產生的多條 Hough edge；要求線段在 notehead 附近終止，排除 barline。對全頁 Hough 漏掉的短 stem，再由未連線 notehead 的左右邊緣進行局部 recovery。
3. `beam`：先找粗實心四邊形，再檢查 stem 遠端端點。第二路徑會測量相鄰 stem 端點間的連續墨跡厚度；細 staff line／slur 不通過。多條平行 beam 會分別輸出並共用 `group_id`。

Xia 標註是稀疏標註，因此只能計算「有標出的目標被找回多少」，不能把未標註的正常音符當成 false positive。評估指令：

```bash
python evaluate_primitives.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia' \
  primitive_output/accuracy_eval
```

在標註最多的 `Beethoven_Op101-01-07.jpeg` 上：

| Primitive | 標註數 | 找回數 | Annotated-target recall |
|---|---:|---:|---:|
| notehead | 58 | 58 | 100.0% |
| stem | 58 | 53 | 91.4% |
| beam | 12 | 11 | 91.7% |

這些數字不是完整資料集 accuracy／precision；仍需搭配 `*_stems.png`、`*_beams.png` 目視檢查。

Flag 預設還會通過 `25-omr` 的 single-stem CNN 二次驗證。只有端點拓撲成立、形狀不是水平的 beam 碎片，而且 CNN 判定為 `n816` 才輸出；`n2`、`n4` 候選會被排除。除錯時可用 `--disable-flag-classifier` 查看純幾何候選。

### Flag 準確率改善方式

目前使用多層判斷，而不是只靠影像輪廓：

1. 先確認 stem 已連到 notehead，而且沒有連到已知 beam。
2. 只搜尋 stem 遠離 notehead 的端點，並要求曲線元件與端點拓撲相連。
3. 排除寬大於高的水平 beam 碎片，再以既有 single-stem CNN 分成 `n2`、`n4`、`n816`；只有 `n816` 保留為 flag。
4. 對 CNN 高信心、但因掃描斷點而無法定位曲線的候選，建立保守的推定框；只有連續多列的粗水平線才當成 beam，細 staff line 不會排除 flag。

Xia 抽樣回歸結果：

| 圖片 | flag 數量 | 檢查結果 |
|---|---:|---|
| `Beethoven_Op090-01-01.jpeg` | 20 | CNN-first 召回版本 |
| `Beethoven_Op090-01-02.jpeg` | 16 | CNN-first 召回版本 |
| `Beethoven_Op090-01-03.jpeg` | 30 | 原先嚴格幾何版本只輸出 2 個 |

以上是模型輸出數量，並非人工 ground truth 的準確率。若仍有漏抓，可先降低 fallback 門檻：

```bash
python detect_primitives.py score.jpeg \
  --flag-classifier-threshold 0.70 \
  --flag-fallback-threshold 0.85
```

`--flag-fallback-threshold` 越低，橘框召回率越高，但也會增加把 slur、文字或漏偵測 beam 當 flag 的風險。

這個流程在推論時只需要圖片。若之後要跨字型、掃描品質與不同出版社繼續提高準確率，下一步應把誤判的 beam、slur、文字與真正 flag crop 蒐集成 hard-negative/hard-positive 集，微調 single-stem classifier；標註只用於訓練與評估，不會成為推論輸入。

## 使用 fine-tuned 25-omr 模型

原始 `seg_net` 輸出：

```text
background / stems_rests / notehead / clefs_keys
```

新的 primitive head 輸出：

```text
background / notehead / stem / beam
```

使用 `.keras` 模型：

```bash
python detect_primitives.py score.jpeg \
  --model primitive_training_output/primitive_seg_net.keras \
  --model-kind primitive \
  --output-dir primitive_output/fine_tuned
```

使用 ONNX：

```bash
python detect_primitives.py score.jpeg \
  --model primitive_training_output/primitive_seg_net.onnx \
  --model-kind primitive \
  --output-dir primitive_output/fine_tuned
```

## Fine-tune 資料檢查

Xia 資料包含 49 張圖片與 49 份 YOLO label。先執行：

```bash
python fine_tune_primitives.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia' \
  --dry-run
```

目前會刻意停止，因為 Xia 現有目標只有：

```text
beamSmall
noteheadBlackInSpaceSmall
noteheadBlackOnLineSmall
stemSmall
```

它們只涵蓋小音符；一般 notehead、stem、beam 未完整標註。若直接 fine-tune，未標註的一般音符會被當成背景。

測試 small-only 資料管線可以明確加入：

```bash
python fine_tune_primitives.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia' \
  --dry-run \
  --allow-small-only
```

## 正式 Fine-tune

先將一般符號補成完整 YOLO boxes，類別名稱至少包含：

```text
noteheadFilled
noteheadHollow
stem
beam
```

安裝訓練依賴：

```bash
python -m pip install -r requirements_primitive_training.txt
```

執行訓練：

```bash
python fine_tune_primitives.py \
  /path/to/completed-dataset \
  --epochs 25 \
  --batch-size 8 \
  --export-onnx
```

程式會：

1. 載入 `25-omr/omr/checkpoints/seg_net/arch.json`。
2. 載入 `weights.h5` backbone 權重。
3. 將最後一層替換成四類 primitive head。
4. 先只訓練新 head。
5. 再解凍 backbone 最後 36 層低學習率微調。
6. 儲存最佳 `.keras`，並可選擇輸出 ONNX。

預設依作品切分，避免相鄰頁資料洩漏：

```text
train: Op090, Op101, Op106
validation: Op109
held-out test: Op110, Op111
```

## Baseline

使用既有 `Beethoven_testing.npy` 快取：

```bash
python detect_primitives.py \
  Beethoven_testing/Beethoven_testing.jpg \
  --legacy-segmentation Beethoven_testing/Beethoven_testing.npy \
  --output-dir primitive_output/baseline
```

目前 baseline 輸出約為：

```text
noteheads=525
stems=357
beams=43
staff_spacing=13.75
```

這些只是候選數量，不等於 ground truth accuracy。正式比較必須用補齊後的標註計算 precision、recall 與 F1。
