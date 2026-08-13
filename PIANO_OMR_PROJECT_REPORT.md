# 鋼琴樂譜 OMR 系統：從圖片辨識 Notehead、Stem、Beam 並輸出 MusicXML

更新日期：2026-08-10

專案性質：既有開源 OMR 系統之整合、鋼琴譜適配與辨識後處理改良

## 一、摘要

本專案以兩個開源專案為基礎：

1. [Bobo1111111/OMR_layout](https://github.com/Bobo1111111/OMR_layout)：原始目標是分析管弦樂總譜頁面，結合 OCR、staff detection 與 GPT，輸出各 staff 對應的樂器、聲部及移調資訊 CSV。
2. [lattellie/25-omr](https://github.com/lattellie/25-omr)：原始目標是將弦樂四重奏 PDF 搭配人工提供的拍號 JSON 轉成 MusicXML，並使用源自 Oemer 的 segmentation model。

我們將兩者整合後，新增一條適合鋼琴譜、只需輸入 JPG／PNG 圖片的流程：

```text
鋼琴樂譜圖片
    ↓
ONNX segmentation
    ↓
notehead / stem / beam / flag 偵測
    ↓
primitive 之間的關係建立
    ↓
overlay 疊圖供人工檢查
    ↓
五線譜與鋼琴左右手推定
    ↓
音高、和弦與基本時值推定
    ↓
MusicXML
```

新流程在推論時不需要人工標註、不需要 `stafflist.pkl`、不需要人工框音符，也不需要 OpenAI API key。標註資料只用於離線評估或未來模型微調。

目前系統已能產生可由 `music21`、MuseScore 等軟體讀取的 MusicXML 初稿；但升降記號、休止符、附點、連結線、tuplet、實際拍號與精確小節邊界仍未完整納入，因此不能將目前輸出視為百分之百正確的完整樂譜轉錄。

## 二、兩個原始專案提供的基礎

### 2.1 OMR_layout 原本的功能

原始 [OMR_layout README](https://github.com/Bobo1111111/OMR_layout/blob/main/README.md) 定義的輸入是單張管弦樂總譜圖片，主要流程包括：

- DocTR OCR 擷取文字與 bounding boxes。
- YOLO 與 OMR segmentation 偵測 staff。
- 過濾 staff 周圍的 OCR 文字。
- 使用 GPT 將文字正規化為樂器名稱。
- 推定 staff group、樂器聲部與移調。
- 最後輸出以 staff 為單位的 CSV。

因此，OMR_layout 提供了全頁圖片處理、staff layout、OCR 與資料整理的專案框架，但原本的主要輸出不是音符 MusicXML。

### 2.2 25-omr 原本的功能

原始 [25-omr README](https://github.com/lattellie/25-omr/blob/main/README.md) 定義的輸入是：

- 弦樂四重奏 PDF。
- 一份人工指定拍號變化、軌數與 clef options 的 JSON。

原始輸出是 MusicXML。專案提供：

- `seg_net` ONNX／Keras segmentation model。
- notehead、stem、beam、rest、clef 等舊版分析函式。
- single-stem rhythm CNN 權重。
- 使用 `music21` 組裝 MusicXML 的既有實作。

原本範例偏向四軌弦樂四重奏，且需要 PDF、設定 JSON 與較多中間資料，不是直接針對鋼琴單頁圖片的簡化流程。

## 三、我們的目標與原專案的差異

| 項目 | OMR_layout 原始版 | 25-omr 原始版 | 本專案新增流程 |
|---|---|---|---|
| 主要輸入 | 管弦樂總譜圖片 | 弦樂四重奏 PDF + JSON | 鋼琴譜 JPG／PNG |
| 主要輸出 | staff／樂器 CSV | MusicXML | overlay + JSON + MusicXML |
| 軌數 | 依總譜而定 | 範例為四軌 | 預設鋼琴左右手兩軌 |
| Clef | 版面／設定處理 | JSON 指定候選 | 預設上方 treble、下方 bass |
| 推論是否需人工標註 | 否 | 不需音符標註，但需設定 JSON | 否，只需圖片 |
| 是否需 OpenAI API | 原始完整流程需要 | 不需要 | 新 MusicXML 路徑不需要 |
| 可視化 | layout／OCR 圖 | 舊版除錯圖 | primitive、stem、beam、XML 音高疊圖 |
| 評估 | staff layout 為主 | 未提供本資料集評估 | 新增稀疏標註 recall 工具 |

本專案的小目標是先提高並穩定三種基礎元件：`notehead`、`stem`、`beam`。只有這三類的關係可靠，後續節奏與 MusicXML 才有可信的結構。

## 四、我們新增與修改的內容

### 4.1 新增獨立 primitive detection 模組

新增 [`primitive_omr/detector.py`](primitive_omr/detector.py)，將舊版大型流程拆成可測試的 primitive detector，輸出：

- notehead bounding box、中心與信心值。
- stem 直線端點、長度、來源與信心值。
- beam polygon、角度、厚度與來源。
- 實驗性 flag bounding box 與分類結果。
- `notehead → stem`、`stem → beam`、`stem → flag` 關係。

這些結果統一寫入 JSON，不再只存在於執行期間的 Python object 或影像 mask。

### 4.2 Notehead 改良

Notehead 不只對 segmentation mask 做簡單 connected components，而是加入：

- 依自動估計的 staff spacing 動態調整尺寸門檻。
- morphology closing，補回掃描造成的斷裂。
- 對黏在一起的垂直和弦 notehead component 進行切分。
- 以面積、長寬、機率與 IoU 去除雜訊和重複框。

因此同一套參數能較穩定地處理不同解析度，而不是使用固定 pixel threshold。

### 4.3 Stem 改良

Stem 是本次最主要的改善項目。新流程加入：

1. 將 segmentation probability 與原始黑白影像的垂直 morphology 結果合併。
2. 使用 Hough line 找垂直線段。
3. 合併同一根粗 stem 造成的左邊、中心與右邊多條 Hough edge，降低重複偵測。
4. 要求 stem 端點接近 notehead，以排除穿過整個 staff 的 barline 或文字直線。
5. 限制純幾何 stem 的最大長度，降低 barline false positive。
6. 對全頁 Hough 遺漏的短 stem，從尚未連線的 notehead 左右邊緣進行局部搜尋，稱為 `note-guided recovery`。
7. 每根 stem 紀錄 `model+geometry`、`geometry` 或 `note-guided-recovery` 來源，方便除錯。

這個做法不是單純降低 threshold；降低全頁 threshold 雖然會增加 recall，也會大量抓到 barline、文字與 slur。局部 recovery 只在已有 notehead 證據的位置放寬條件。

### 4.4 Beam 改良

Beam 偵測使用兩條互補路徑：

- 形狀路徑：尋找粗、實心、接近水平或斜向的四邊形，再驗證是否靠近 stem 遠端端點。
- 關係路徑：若多根 stem 的遠端端點成群，檢查兩端之間是否有連續且具有厚度的黑色 bridge。

另外加入：

- 計算「最長連續垂直墨跡」，而不是把數條細 staff line 的墨跡相加，避免把 staff／slur 誤判成 beam。
- 多條平行 beam 分帶偵測；兩層 beam 會分成兩個 band，而不是只得到一個大框。
- 以 `group_id` 與 `stem_to_beam` 保存 beam group 和 stem 的關係。
- 將 shape detection 與 multi-stem recovery 的重複結果去除。

### 4.5 Flag 的實驗性處理

Flag 不是目前主要評估目標，但系統保留兩種路徑：

- stem 遠端幾何與拓撲搜尋。
- 重用 25-omr 的 single-stem CNN，將候選分成 `n2`、`n4`、`n816`，只接受高信心的 `n816`。

Flag 目前仍比 notehead、stem、beam 不穩定，因此報告與定量結果不把 flag 當成已解決項目。

### 4.6 大圖分塊推論與快取

新增 [`primitive_omr/inference.py`](primitive_omr/inference.py)：

- 自動將不同大小的頁面縮放到合理像素數。
- 對 25-omr ONNX model 執行 overlapping tiled inference。
- 使用 Hanning weight 混合相鄰 tile，降低 tile 邊界斷裂。
- 將 segmentation probability 存成 `.cache/*_segmentation.npz`。
- 第二次執行相同圖片和輸出目錄時直接讀快取。
- 支援原始四 channel model，以及未來 fine-tuned 的四／五 channel primitive model。

### 4.7 新增多種 overlay

新增 [`detect_primitives.py`](detect_primitives.py) 作為 primitive CLI。每張圖片會輸出：

- `*_overlay.png`：所有 primitive 疊圖。
- `*_stems.png`：突出 stem 來源與尚未連線的 notehead。
- `*_beams.png`：突出 beam、recovery beam 與 flag。
- `*.json`：完整座標、信心值與關係。

Overlay 的用途不只是展示，而是讓研究者知道錯誤來自 segmentation、幾何 recovery，或 primitive 關係組裝。

### 4.8 鋼琴 staff、音高與 MusicXML

新增 [`primitive_omr/musicxml.py`](primitive_omr/musicxml.py) 與 [`image_to_musicxml.py`](image_to_musicxml.py)：

- 直接從原圖的長水平線自動找出每組五線譜。
- 排除由密集 notehead／beam 形成的假五線譜候選。
- 預設將相鄰 staff 配成鋼琴大譜表。
- 上方 staff 使用 treble clef，下方使用 bass clef。
- 依 notehead 的 y 座標和 staff spacing 換算自然音高。
- 同一根 stem 連到多個 notehead 時輸出 chord。
- 依 notehead 填色、stem、beam 層數與 flag 層數估算基本時值。
- 產生帶 staff 編號和推定音高文字的 `*_musicxml_overlay.png`。
- 使用 `music21` 輸出可解析的 `.musicxml`。
- 將所有假設與尚未辨識的資訊寫入 `*_conversion.json`。

### 4.9 評估與 fine-tune 工具

新增：

- [`evaluate_primitives.py`](evaluate_primitives.py)：將輸出和 Xia 的 YOLO boxes 比對。
- [`fine_tune_primitives.py`](fine_tune_primitives.py)：準備替換 25-omr segmentation head，目標 channel 為 `background / notehead / stem / beam`。
- [`tests/test_primitive_detector.py`](tests/test_primitive_detector.py)：primitive 幾何與關係測試。
- [`tests/test_musicxml_conversion.py`](tests/test_musicxml_conversion.py)：staff、pitch、rhythm 與 MusicXML parse 測試。

重要的是，fine-tune 程式目前是「已完成資料驗證與訓練管線」，不是「已經用 Xia 全頁標註訓練完成的新模型」。原因見第七節。

### 4.10 新增、修改與沿用程式碼對照

為避免將上游程式誤列為本組成果，以下依目前 Git working tree 與兩個 upstream 專案進行分類。

#### A. 本組全新新增：正式新流程

| 程式 | 主要新增內容 | 是否為建議入口 |
|---|---|---|
| `primitive_omr/detector.py` | notehead、stem、beam、flag 偵測、recovery、關係建立及 overlay | 核心模組 |
| `primitive_omr/inference.py` | 大圖縮放、overlapping tiled ONNX／Keras inference、tile 混合、座標縮放 | 核心模組 |
| `primitive_omr/rhythm_classifier.py` | 包裝並重用 25-omr single-stem CNN，輸出 flag／rhythm 信心值 | 輔助模組 |
| `primitive_omr/musicxml.py` | 五線譜偵測、鋼琴左右手、音高、和弦、基本時值、MusicXML 與最終 overlay | 核心模組 |
| `detect_primitives.py` | 單張／批次圖片轉 primitive JSON 與多種 overlay 的 CLI | 是，primitive 檢查入口 |
| `image_to_musicxml.py` | 串接圖片、primitive、overlay、conversion JSON 與 MusicXML | 是，完整正式入口 |
| `evaluate_primitives.py` | 與 Xia 稀疏 YOLO 標註比對 annotated-target recall | 評估入口 |
| `fine_tune_primitives.py` | 建立 primitive mask、切 patch、替換 segmentation head、訓練／ONNX export | 未來訓練入口 |
| `tests/test_primitive_detector.py` | primitive 幾何與關係回歸測試 | 測試 |
| `tests/test_musicxml_conversion.py` | staff、pitch、duration 與 MusicXML parse 測試 | 測試 |

#### B. 本組全新新增：早期整合 prototype

| 程式 | 用途 | 現在是否建議使用 |
|---|---|---|
| `run_piano_omr.py` | 將 25-omr 舊版流程改為固定鋼琴兩軌的早期嘗試 | 否，依賴舊中間物件 |
| `run_combined_omr.py` | 嘗試結合 layout boxes、stafflist 與 25-omr MusicXML | 否，依賴 `stafflist.pkl` |
| `Beethoven_testing/export_direct_noteheads_to_xml.py` | 直接把早期 notehead 結果送進舊版 XML pipeline | 否，僅保留實驗紀錄 |

這些 prototype 幫助確認 25-omr 可以輸出鋼琴雙 part XML，但正式版改用 `image_to_musicxml.py`，因為正式版只需圖片、輸出結構固定，並且每一層都可單獨測試。

#### C. 本組新增：環境、說明與重現文件

| 檔案 | 用途 |
|---|---|
| `requirements_piano_omr.txt` | 經實測的 Python 3.11 inference 依賴 |
| `requirements_primitive_training.txt` | TensorFlow／ONNX fine-tune 額外依賴 |
| `PRIMITIVE_DETECTION.md` | primitive 演算法、門檻與稀疏標註評估 |
| `IMAGE_TO_MUSICXML.md` | 圖片到 overlay／MusicXML 操作說明 |
| `JPG_TO_MUSICXML_GUIDE.md` | 早期直接執行 25-omr 的實驗紀錄 |
| `PIANO_OMR_PROJECT_REPORT.md` | 本份老師匯報與重現報告 |

#### D. 對既有程式的實際修改

目前正式新流程採取「新增 adapter／module」而不是直接覆寫 upstream，因此核心既有程式的修改很少：

| 既有檔案 | 目前修改 | 是否影響正式新流程 |
|---|---|---|
| `25-omr/string_dataset/piecesToRun.json` | 將舊 demo 曲目名稱由 `beethoven1` 改成實驗曲目設定 | 否；`image_to_musicxml.py` 不讀此檔 |
| OMR_layout 原有 OCR／layout scripts | working tree 顯示多個檔案有換行格式差異，但忽略行尾空白後沒有本次 primitive 核心演算法差異 | 否 |

`JPG_TO_MUSICXML_GUIDE.md` 記錄過 CPU provider、`NoneType` 保護與 Windows path 等早期嘗試；但以目前 Git 狀態核對，這些變更並未保留在 `25-omr` 核心檔案中，因此不能列為目前版本已套用的正式修改。

#### E. 直接沿用、沒有修改的 upstream 資產

| 上游資產 | 本組使用方式 |
|---|---|
| `25-omr/omr/checkpoints/seg_net/model.onnx` | 產生 notehead 與 stem/rest probability maps；權重未修改 |
| `25-omr/training/stemupImg32x32_best.pth` | stem-up rhythm／flag 輔助分類；權重未修改 |
| `25-omr/training/stemdownImg32x32_best.pth` | stem-down rhythm／flag 輔助分類；權重未修改 |
| `25-omr/pdf2musicXML.py` | 作為舊版 MusicXML 邏輯與 prototype 參考；正式新入口不直接執行它 |
| OMR_layout OCR／staff layout pipeline | 保留作為完整總譜版面與樂器分析路徑；新 primitive CLI 可獨立執行 |

因此，本組所稱的「改得更準確」主要是新 `primitive_omr` 後處理、幾何 recovery、false-positive 排除及 primitive relations，不是宣稱已重新訓練或修改上游 ONNX／CNN 權重。

## 五、目前建議使用的完整架構

```text
                         ┌────────────────────────┐
                         │ 25-omr seg_net ONNX    │
                         │ probabilities          │
                         └───────────┬────────────┘
                                     │
JPG / PNG ── resize + tiled inference ── primitive detector
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
          noteheads                stems                  beams
              │                      │                      │
              └──── note→stem links ┴── stem→beam links ──┘
                                     │
                             primitive JSON
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                  overlay QA              staff/pitch/rhythm
                                                   │
                                             MusicXML
```

舊的 [`run_piano_omr.py`](run_piano_omr.py) 與 [`run_combined_omr.py`](run_combined_omr.py) 是整合 25-omr 大型舊流程時的 prototype，仍依賴較多舊物件與中間檔。現在建議正式使用 `image_to_musicxml.py`，因為入口、輸出和假設都較清楚，也能測試每一層。

## 六、安裝方式

### 6.1 取得專案

```bash
git clone https://github.com/Bobo1111111/OMR_layout.git
cd OMR_layout
```

如果修改版 repository 尚未包含 `25-omr/`，再執行：

```bash
git clone https://github.com/lattellie/25-omr.git 25-omr
```

以下指令假設使用的 branch 已包含本報告列出的新增檔案，例如：

```text
image_to_musicxml.py
detect_primitives.py
primitive_omr/
requirements_piano_omr.txt
```

若 GitHub `main` 尚未合併這些檔案，必須先 checkout 本組的實作 branch 或取得本次修改版；只 clone 兩個原始 upstream 並不會自動得到這些新增功能。

### 6.2 建立 Conda 環境

本次驗證環境是 macOS、Python 3.11.15：

```bash
conda create -n omr25-311 python=3.11 -y
conda activate omr25-311
python -m pip install --upgrade pip
python -m pip install -r requirements_piano_omr.txt
```

主要版本：

| 套件 | 驗證版本 | 用途 |
|---|---:|---|
| numpy | 1.26.4 | array 與 mask |
| opencv-python-headless | 4.11.0.86 | 圖像與幾何處理 |
| onnxruntime | 1.18.0 | segmentation inference |
| torch | 2.7.1 | single-stem rhythm CNN |
| scipy | 1.16.0 | staff-line peak detection |
| music21 | 9.7.1 | MusicXML 組裝與驗證 |

確認環境：

```bash
python -c "import torch, cv2, onnxruntime, music21, scipy; print('環境安裝成功')"
```

模型檔應存在：

```text
25-omr/omr/checkpoints/seg_net/model.onnx
25-omr/training/stemupImg32x32_best.pth
25-omr/training/stemdownImg32x32_best.pth
```

## 七、如何執行

### 7.1 一個指令：圖片 → overlay → MusicXML

```bash
conda activate omr25-311
cd /path/to/OMR_layout

python image_to_musicxml.py \
  /path/to/score.jpg \
  --output-dir musicxml_output/score
```

以本次 Xia Beethoven 圖片為例：

```bash
python image_to_musicxml.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images/Beethoven_Op101-01-07.jpeg' \
  --output-dir musicxml_output/Beethoven_Op101-01-07
```

第一次會跑 ONNX segmentation；同一圖片再次使用相同輸出目錄時會讀取 `.cache`。若模型或程式更新後要強制重新推論：

```bash
python image_to_musicxml.py score.jpg \
  --output-dir musicxml_output/score \
  --force
```

預設模式是鋼琴雙 staff。其他模式：

```bash
python image_to_musicxml.py score.jpg --staff-mode treble
python image_to_musicxml.py score.jpg --staff-mode bass
```

目前拍號尚未自動辨識，可提供 fallback：

```bash
python image_to_musicxml.py score.jpg --time-signature 3/4
```

### 7.2 只檢查 notehead、stem、beam overlay

```bash
python detect_primitives.py \
  /path/to/score.jpg \
  --output-dir primitive_output/score
```

可調整主要門檻：

```bash
python detect_primitives.py score.jpg \
  --notehead-threshold 0.28 \
  --stem-threshold 0.22 \
  --output-dir primitive_output/score
```

除錯 flag 時，可停用 CNN，只看幾何候選：

```bash
python detect_primitives.py score.jpg \
  --disable-flag-classifier \
  --output-dir primitive_output/geometry_only
```

### 7.3 批次處理資料夾

```bash
python image_to_musicxml.py \
  /path/to/images \
  --output-dir musicxml_output/batch
```

先測前三張：

```bash
python image_to_musicxml.py \
  /path/to/images \
  --output-dir musicxml_output/batch \
  --limit 3
```

若圖片位於子資料夾：

```bash
python image_to_musicxml.py \
  /path/to/dataset \
  --recursive \
  --output-dir musicxml_output/batch
```

### 7.4 執行測試

```bash
python -m unittest discover -s tests -v
```

目前共 8 項測試，涵蓋：

- staff spacing。
- primitive detection 與關係。
- Hough stem edge 合併。
- beam bridge 排除細 staff line。
- direct flag segmentation channel。
- treble／bass staff 與 pitch。
- beam 對應八分音符。
- MusicXML 寫出後重新解析。

### 7.5 稀疏標註評估

先產生 prediction：

```bash
python detect_primitives.py \
  /path/to/Xia/images \
  --output-dir primitive_output/xia
```

再評估：

```bash
python evaluate_primitives.py \
  /path/to/Xia \
  primitive_output/xia
```

### 7.6 Fine-tune 資料檢查

先只檢查資料，不載入 TensorFlow：

```bash
python fine_tune_primitives.py \
  /path/to/Xia \
  --dry-run
```

Xia 目前只對 small primitive 提供相關類別，因此程式預設會停止，避免把未標註的一般音符當成背景。若只是驗證訓練資料管線：

```bash
python fine_tune_primitives.py \
  /path/to/Xia \
  --dry-run \
  --allow-small-only
```

只有在一般尺寸 notehead、stem、beam 都完成全頁標註後，才建議正式訓練：

```bash
python -m pip install -r requirements_primitive_training.txt

python fine_tune_primitives.py \
  /path/to/completed-dataset \
  --epochs 25 \
  --batch-size 8 \
  --export-onnx
```

## 八、輸出檔案與閱讀方式

假設輸入是 `score.jpg`：

```text
musicxml_output/score/
├── score.json
├── score_overlay.png
├── score_stems.png
├── score_beams.png
├── score_musicxml_overlay.png
├── score_conversion.json
├── score.musicxml
└── .cache/
    └── score_segmentation.npz
```

| 檔案 | 用途 |
|---|---|
| `score_overlay.png` | 同時看 notehead、stem、beam、flag |
| `score_stems.png` | 檢查 stem 來源與漏接 notehead |
| `score_beams.png` | 檢查 beam 層數、recovery 與 flag |
| `score_musicxml_overlay.png` | 檢查 staff 分配及即將寫入 XML 的音高 |
| `score.json` | primitive 座標、信心值與關係 |
| `score_conversion.json` | staff、note event、時值來源、輸出路徑與警告 |
| `score.musicxml` | 可交給 MuseScore／music21 的結果 |

建議檢查順序：

1. 先看 `*_stems.png` 和 `*_beams.png`，確認 primitive。
2. 再看 `*_musicxml_overlay.png`，確認五線譜與音高位置。
3. 最後才用 MuseScore 打開 MusicXML。

## 九、資料與評估結果

### 9.1 Xia 資料狀況

目前 Xia 資料包含：

- 49 張圖片。
- 49 份 YOLO label。
- 六首 Beethoven 作品：Op090、Op101、Op106、Op109、Op110、Op111。

與本階段三類 primitive 對應的標註數：

| Target | 標註數 |
|---|---:|
| notehead | 128 |
| stem | 128 |
| beam | 88 |

這些類別全部是 `notehead*Small`、`stemSmall`、`beamSmall`，不是完整頁面的所有一般音符。另有 9,494 個其他類別 annotation 未映射到本階段三類 target。

### 9.2 稀疏標註回歸測試

在標註較多的 `Beethoven_Op101-01-07.jpeg` 上：

| Primitive | 已標註 | 找回 | Annotated-target recall | Mean match quality |
|---|---:|---:|---:|---:|
| notehead | 58 | 58 | 100.0% | 0.7186 |
| stem | 58 | 53 | 91.4% | 0.9262 |
| beam | 12 | 11 | 91.7% | 0.5746 |

這裡只能稱為「已標註目標的 recall」，不能稱為完整 accuracy、precision 或 F1。原因是資料沒有標出頁面上所有一般 notehead、stem 和 beam；未標註但正確偵測到的普通音符不能被當成 false positive。

目前也沒有保存同一評估條件下的原始 baseline recall，因此本報告不虛構「提升了多少百分點」。可確認的改進是新增的去重、barline 排除、短 stem recovery、beam 厚度與關係驗證，以及上述當前回歸結果。

### 9.3 完整圖片到 XML 實測

在 `Beethoven_Op101-01-07.jpeg` 上：

| 項目 | 結果 |
|---|---:|
| 自動偵測 staff | 12 |
| note／chord events | 412 |
| 匯出 pitches | 548 |
| MusicXML parts | 2 |

輸出的 MusicXML 已通過：

```bash
xmllint --noout \
  musicxml_output/Beethoven_Op101-01-07/Beethoven_Op101-01-07.musicxml
```

並已使用 `music21.converter.parse()` 重新讀取成功。

這項驗證代表 XML 結構有效，不代表每一個音高與節奏都已等同原譜。

## 十、MusicXML 的目前推定規則

| 影像證據 | 匯出時值 |
|---|---|
| 一層 beam／flag | 八分音符 |
| 兩層 beam／flag | 十六分音符 |
| 三層 beam／flag | 三十二分音符 |
| 實心 notehead + stem | 四分音符 |
| 空心 notehead + stem | 二分音符 |
| 空心 notehead、無 stem | 全音符 |

音高規則：

- Treble staff 最下線為 E4。
- Bass staff 最下線為 G2。
- 每半個 staff spacing 移動一個 diatonic step。
- 目前只匯出自然音；升降記號尚未套用。

鋼琴規則：

- 第 1、3、5…個 staff 為右手 treble。
- 第 2、4、6…個 staff 為左手 bass。
- 每兩個 staff 組成一個 system。

以上假設都會在 `*_conversion.json` 中留下 warning，避免把推定值誤當 ground truth。

## 十一、目前限制

目前尚未完整辨識：

- key signature。
- 臨時升降記號。
- 圖片中的 clef change。
- 圖片中的 time signature。
- rest。
- dotted note。
- tie 與 slur 的語意區分。
- tuplet。
- 多 voice 與 cross-staff voice。
- barline 與精確小節邊界。
- 跨頁連續性。

因此目前 MusicXML 的定位是「可檢查、可解析的第一版轉錄」，最適合用於：

- 驗證 primitive detection。
- 將偵測結果快速送入 MuseScore 人工校正。
- 建立後續 accidental、rest、barline 模型的輸入結構。

## 十二、下一步工作

建議依下列順序繼續：

1. 完整標註一般大小的 notehead、stem、beam，才能正式 fine-tune 並計算 precision／recall／F1。
2. 加入 barline 與 rest，使每小節時值成立。
3. 加入 accidental 與 key signature，修正實際音高。
4. 加入 dot、tie、tuplet，修正節奏。
5. 辨識 clef 與 time signature，移除目前的 fallback。
6. 建立以作品為單位的 train／validation／test split，避免相鄰頁資料洩漏。
7. 保存每個版本的 prediction 和 evaluation JSON，才能量化每次改動相對 baseline 的提升。

## 十三、結論

本專案不是重新從零訓練一套 OMR，而是利用兩個既有專案的互補能力：

- OMR_layout 提供全頁圖片、staff layout 與專案 pipeline 基礎。
- 25-omr 提供 segmentation model、single-stem CNN 與 MusicXML 經驗。

我們新增的主要貢獻是將流程改成適合鋼琴譜的兩軌架構，建立獨立且可測試的 notehead／stem／beam detector，加入以音符關係為核心的 stem／beam recovery 和 false-positive 排除，產生多種 overlay，並將同一份 primitive JSON 轉換成可解析的 MusicXML。

目前三類 primitive 在稀疏標註頁面上已有可重現的 recall 結果，完整圖片到 MusicXML 也已跑通；但資料標註不完整，因此我們保留清楚的限制說明，並把正式 fine-tune 與完整樂譜語意辨識列為下一階段。

## 參考資料

1. Bobo1111111, [OMR_layout](https://github.com/Bobo1111111/OMR_layout), GitHub.
2. lattellie, [25-omr](https://github.com/lattellie/25-omr), GitHub.
3. BreezeWhite, [Oemer](https://github.com/BreezeWhite/oemer), segmentation model upstream referenced by 25-omr.
4. [`PRIMITIVE_DETECTION.md`](PRIMITIVE_DETECTION.md), 本專案 primitive 技術與評估說明。
5. [`IMAGE_TO_MUSICXML.md`](IMAGE_TO_MUSICXML.md), 本專案圖片到 MusicXML 操作說明。
