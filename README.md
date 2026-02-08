# Weather Draw Island - Feedback Retraining Pipeline

フィードバックデータを使ってモデルを再学習するPythonスクリプト集

---

## 📁 ファイル構成

```
python_project/
├── feedback_export.json          # Unity からエクスポート（Eキー）
├── clean_feedback.py             # データクリーニング
├── retrain_with_feedback.py      # 再学習（Fine-tuning）
├── evaluate_model.py             # モデル評価・比較
├── train_model.py                # （既存）ゼロから学習
│
├── weather_model_v6.pth          # 既存モデル（98.89%）
├── weather_model_retrained.pth   # 再学習後モデル（生成される）
├── weather_model_retrained.onnx  # Unity用（生成される）
│
└── ../Assets/MyFolder/           # Unityプロジェクト（相対パス）
    └── Img_Output/               # 描画画像
        └── drawing_XXXX.png
```

---

## 🚀 使い方（3ステップ）

### Step 1: データクリーニング

```bash
python clean_feedback.py
```

**やること:**
- `feedback_export.json`を読み込み
- 画像存在チェック
- ピクセルカバレッジ: 5-23%
- 信頼度Gap: >= 2.0（警告のみ）
- 重複除外（最新を優先）

**出力:**
- `feedback_cleaned.json`（クリーニング済み）
- 統計レポート（コンソール）

**例:**
```
Total:              45
Image not found:    2
Low coverage:       3
High coverage:      1
Low confidence:     5 (not excluded)
Duplicates:         8
Valid:              31
Removal rate:       31.1%

Class distribution (cleaned):
  sun:  10
  moon: 12
  rain: 9
```

---

### Step 2: 再学習（Fine-tuning）

```bash
python retrain_with_feedback.py
```

**やること:**
- 既存データセット（`train/sun/`, `train/moon/`, `train/rain/`）読み込み
- フィードバックデータ統合
- 既存モデル（`weather_model_v6.pth`）読み込み
- Fine-tuning実行（低学習率: 0.0001）
- Early Stopping（patience=15）
- ONNX変換

**出力:**
- `weather_model_retrained.pth`（PyTorch）
- `weather_model_retrained.onnx`（Unity用）

**パラメータ:**
- Epochs: 50（最大）
- Learning Rate: 0.0001（Fine-tuning用）
- Batch Size: 16
- Train/Val Split: 80/20

**例:**
```
Loaded 500 images from train
Loaded 31 images from feedback
Combined dataset size: 531
Train size: 424, Val size: 107

Epoch [1/50] Train Loss: 0.1234, Train Acc: 96.50%, Val Acc: 97.20%
  → Best model saved! (Val Acc: 97.20%)
...
Early stopping triggered (no improvement for 15 epochs)

Training finished. Best validation accuracy: 99.07%
ONNX model exported: weather_model_retrained.onnx
```

---

### Step 3: 評価・比較

```bash
python evaluate_model.py
```

**やること:**
- Before（`weather_model_v6.pth`）を評価
- After（`weather_model_retrained.pth`）を評価
- 混同行列を画像出力
- Before/After比較レポート

**出力:**
- `confusion_matrix_before.png`
- `confusion_matrix_after.png`
- 比較レポート（コンソール）

**例:**
```
[BEFORE] Evaluating original model...
Accuracy: 98.89%

[AFTER] Evaluating retrained model...
Accuracy: 99.07%

Comparison Summary
==================
BEFORE: 98.89%
AFTER:  99.07%
Improvement: +0.18% ✓
```

---

## ⚙️ 設定（必要に応じて変更）

### `clean_feedback.py`

```python
FEEDBACK_JSON = 'feedback_export.json'
UNITY_PROJECT_PATH = '../Assets/MyFolder'  # Unityプロジェクトへの相対パス
MIN_COVERAGE = 0.05  # 5%
MAX_COVERAGE = 0.23  # 23%
MIN_CONFIDENCE_GAP = 2.0
```

### `retrain_with_feedback.py`

```python
FEEDBACK_JSON = 'feedback_cleaned.json'
UNITY_PROJECT_PATH = '../Assets/MyFolder'
EXISTING_MODEL = 'weather_model_v6.pth'
```

---

## 📊 データ品質管理（2段階）

### Stage 1: 自動除外（clean_feedback.py）

**除外対象:**
- 画像が見つからない
- ピクセルカバレッジ < 5%（描画少なすぎ）
- ピクセルカバレッジ > 23%（塗りつぶし）
- 重複画像（同一パスは最新のみ保持）

### Stage 2: 警告のみ（除外しない）

**警告対象:**
- 信頼度Gap < 2.0（モデルが迷っている）
- 人間が最終判断すべき

---

## 🔄 Unity への適用

### 1. ONNXファイルをUnityにコピー

```bash
# 生成されたONNXファイルをUnityプロジェクトにコピー
cp weather_model_retrained.onnx ../Assets/MyFolder/Models/
```

### 2. Unity Editorで差し替え

1. `ONNXInferenceManager`のInspector
2. `Model Asset`に`weather_model_retrained.onnx`をドラッグ
3. Play実行
4. 精度向上を確認

---

## 🐛 トラブルシューティング

### 画像が見つからない（Image not found）

**原因:** `UNITY_PROJECT_PATH`が間違っている

**解決:**
```python
# clean_feedback.py の2行目を修正
UNITY_PROJECT_PATH = '../Assets/MyFolder'  # 相対パス確認
```

### 既存データがない（'train' folder not found）

**原因:** `train/sun/`, `train/moon/`, `train/rain/`フォルダがない

**解決:**
- フィードバックデータのみで学習可能
- 警告は出るが続行される

### GPUが使えない（Using device: cpu）

**原因:** PyTorch GPUサポートなし

**解決:**
- CPUでも動作可能（遅いだけ）
- GPU版PyTorch再インストール推奨

### モデルファイルが見つからない

**エラー:** `weather_model_v6.pth not found`

**解決:**
- Fine-tuningスキップ
- ゼロから学習される（警告表示）

---

## 📈 期待される効果

### Before（既存モデル）
- 精度: 98.89%
- 問題: 実際のユーザー描画で失敗

### After（再学習後）
- 精度: 99%+（期待値）
- 効果: ユーザー特有の描き方に対応

### 具体例
- Before: 太陽を雨と誤認識
- After: ユーザーの描いた太陽パターンを学習済み → 正解

---

## 🔁 継続的改善サイクル

```
1. Unity でゲームプレイ
   ↓
2. フィードバック収集（SQLite）
   ↓
3. JSON エクスポート（Eキー）
   ↓
4. clean_feedback.py
   ↓
5. retrain_with_feedback.py
   ↓
6. evaluate_model.py
   ↓
7. Unity へ適用
   ↓
8. 精度向上確認
   ↓
1. に戻る（繰り返し）
```

---

## 📝 注意事項

### データ量について
- 最低30件以上推奨
- 各クラス最低10件ずつ
- 少なすぎる場合は精度低下の可能性

### クラスバランス
- sun/moon/rainの件数が均等に近いほど良い
- 偏りがある場合は追加収集推奨

### 学習時間
- CPU: 10-30分
- GPU: 3-10分
- データ量による

---

## 🎯 ポートフォリオアピールポイント

### 技術的深度
✅ MLOpsの実践（データ品質管理）  
✅ Unity-Python連携  
✅ SQLiteデータベース設計  
✅ リアルタイムフィルタリング  
✅ 継続的学習パイプライン  

### 実用性
✅ 実際に動作するフィードバックループ  
✅ 段階的品質管理（2段階フィルタ）  
✅ 人間介入の効率化  

---

**作成者:** Taro  
**最終更新:** 2026-02-09
