# 圖片 → overlay 疊圖 → MusicXML

## 直接執行

```bash
conda activate omr25-311
cd /Users/shih/Desktop/OMR_layout

python image_to_musicxml.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images/Beethoven_Op101-01-07.jpeg' \
  --output-dir musicxml_output/Beethoven_Op101-01-07
```

這個指令只需要圖片，不需要 YOLO 標註、`stafflist.pkl` 或人工指定音符位置。

預設把相鄰兩個 staff 當成鋼琴的大譜表：上方使用 treble clef、下方使用 bass clef。單一高音或低音譜表可改用：

```bash
python image_to_musicxml.py score.jpg --staff-mode treble
python image_to_musicxml.py score.jpg --staff-mode bass
```

如果已知拍號，可以給 fallback 拍號：

```bash
python image_to_musicxml.py score.jpg --time-signature 3/4
```

## 完整資料流

```text
輸入 JPG/PNG
    ↓
25-omr ONNX segmentation
    ↓
notehead / stem / beam / flag 幾何偵測與關聯
    ↓
primitive JSON + primitive overlay
    ↓
自動偵測五線譜，分配 treble / bass staff
    ↓
notehead 的 y 座標換算成音高
    ↓
stem + notehead 填色 + beam/flag 層數換算時值
    ↓
MusicXML + 帶 staff 與音高文字的最終 overlay
```

時值規則如下：

| 影像關係 | 匯出時值 |
|---|---|
| 1 層 beam 或 flag | 八分音符 |
| 2 層 beam 或 flag | 十六分音符 |
| 3 層 beam 或 flag | 三十二分音符 |
| 實心 notehead + stem | 四分音符 |
| 空心 notehead + stem | 二分音符 |
| 空心 notehead、無 stem | 全音符 |

同一根 stem 連到多個 notehead 時會匯出為 chord。模型快取存於輸出目錄的 `.cache`，用同一個輸出目錄重跑時不會再次執行 ONNX segmentation；加入 `--force` 才會重算。

## 輸出檔案

對 `score.jpg` 執行後會得到：

```text
musicxml_output/
├── score.json                  # primitive 與關聯
├── score_overlay.png          # notehead/stem/beam/flag 疊圖
├── score_stems.png             # stem 診斷圖
├── score_beams.png             # beam/flag 診斷圖
├── score_musicxml_overlay.png  # staff、primitive、推定音高的最終疊圖
├── score_conversion.json       # staff、事件、假設與警告
├── score.musicxml              # 最終 MusicXML
└── .cache/
    └── score_segmentation.npz
```

建議先看 `score_musicxml_overlay.png`：綠／橘水平線是系統找到的 treble／bass staff，紅字是即將寫入 XML 的推定音高。確認對齊後，再用 MuseScore、Dorico 或 `music21` 開啟 `score.musicxml`。

## 實際驗證

`Beethoven_Op101-01-07.jpeg` 的完整流程結果：

| 項目 | 數量 |
|---|---:|
| staff | 12 |
| 音符／和弦事件 | 412 |
| 匯出的 pitches | 548 |
| MusicXML parts | 2 |

XML 已通過：

```bash
xmllint --noout musicxml_output/Beethoven_Op101-01-07/Beethoven_Op101-01-07.musicxml
```

也已使用 `music21.converter.parse()` 重新載入，確認 XML 結構可解析。

## 目前準確度邊界

現在輸出的 MusicXML 是「可解析的 primitive-based 初稿」，還不是完整的樂譜語意轉錄。系統尚未把以下符號納入新轉換層：

- key signature 與臨時升降記號
- 圖片內實際的 time signature
- rest、附點、tie、tuplet 與多 voice
- barline 與精確小節邊界
- 圖片中的 clef change

因此目前音名使用 natural pitch；拍號由 `--time-signature` 提供；小節先依系統換行與累積時值建立。這些限制也會寫入每張圖的 `*_conversion.json`。

若目標是讓 XML 可直接演奏並接近原譜，下一階段優先順序應為：

1. barline + rest，先讓小節總時值成立。
2. accidental + key signature，修正實際音高。
3. tie + dot + tuplet，修正跨小節與非二分節奏。
4. clef/time-signature classifier，移除目前的 fallback 假設。

## 批次轉換

```bash
python image_to_musicxml.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images' \
  --output-dir musicxml_output/xia
```

先測前三張：

```bash
python image_to_musicxml.py \
  '/Users/shih/Documents/交大/114-2/HCI/中研院/Xia/images' \
  --output-dir musicxml_output/xia \
  --limit 3
```
