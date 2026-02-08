import json
import os
from PIL import Image
import numpy as np
from collections import defaultdict

# ========== 設定 ==========
FEEDBACK_JSON = 'feedback_export.json'
UNITY_PROJECT_PATH = r'C:\UnityProjects\Weather Draw Islands\Assets\MyFolder'  # Unityプロジェクトへの相対パス（必要に応じて変更）
MIN_COVERAGE = 0.05  # 5%
MAX_COVERAGE = 0.23  # 23%
MIN_CONFIDENCE_GAP = 2.0

# ========== ピクセルカバレッジ計算 ==========
def calculate_coverage(img_path):
    """
    画像のピクセルカバレッジを計算
    背景: 黒（0）、描画: 白（255）
    """
    try:
        img = Image.open(img_path).convert('L')
        pixels = np.array(img)
        
        # 白ピクセル（描画部分）をカウント
        total_pixels = pixels.size
        drawing_pixels = np.sum(pixels >= 128)  # 閾値128以上を「白」とみなす
        
        coverage = drawing_pixels / total_pixels
        return coverage
    except Exception as e:
        print(f"  Error calculating coverage for {img_path}: {e}")
        return None

# ========== 信頼度ギャップ計算 ==========
def calculate_confidence_gap(scores):
    """
    1位と2位のスコア差を計算
    """
    sorted_scores = sorted(scores, reverse=True)
    gap = sorted_scores[0] - sorted_scores[1]
    return gap

# ========== メイン処理 ==========
def clean_feedback():
    print("=" * 60)
    print("Feedback Data Cleaning")
    print("=" * 60)
    
    # JSONロード
    if not os.path.exists(FEEDBACK_JSON):
        print(f"Error: {FEEDBACK_JSON} not found!")
        return
    
    with open(FEEDBACK_JSON, 'r') as f:
        data = json.load(f)
    
    feedbacks = data['feedbacks']
    print(f"\nTotal feedbacks in JSON: {len(feedbacks)}")
    
    # 統計
    stats = {
        'total': len(feedbacks),
        'image_not_found': 0,
        'low_coverage': 0,
        'high_coverage': 0,
        'low_confidence': 0,
        'duplicates': 0,
        'valid': 0
    }
    
    # 重複除外用（同一画像パスは最新のみ保持）
    unique_feedbacks = {}
    
    cleaned_data = []
    
    for fb in feedbacks:
        img_path = fb['image_path']
        
        # Unity プロジェクトの絶対パス構築
        # 例: "Img_Output/drawing_0006.png" → "../Assets/MyFolder/Img_Output/drawing_0006.png"
        full_img_path = os.path.join(UNITY_PROJECT_PATH, img_path)
        
        # 1. 画像存在チェック
        if not os.path.exists(full_img_path):
            print(f"  ✗ Image not found: {full_img_path}")
            stats['image_not_found'] += 1
            continue
        
        # 2. ピクセルカバレッジチェック
        coverage = calculate_coverage(full_img_path)
        if coverage is None:
            stats['image_not_found'] += 1
            continue
        
        if coverage < MIN_COVERAGE:
            print(f"  ✗ Low coverage ({coverage:.2%}): {img_path}")
            stats['low_coverage'] += 1
            continue
        
        if coverage > MAX_COVERAGE:
            print(f"  ✗ High coverage ({coverage:.2%}): {img_path}")
            stats['high_coverage'] += 1
            continue
        
        # 3. 信頼度ギャップチェック（警告のみ、除外しない）
        scores = [
            fb['predicted_score_sun'],
            fb['predicted_score_moon'],
            fb['predicted_score_rain']
        ]
        confidence_gap = calculate_confidence_gap(scores)
        
        if confidence_gap < MIN_CONFIDENCE_GAP:
            print(f"  ⚠ Low confidence (gap={confidence_gap:.2f}): {img_path}")
            stats['low_confidence'] += 1
            # 警告だけで続行（除外しない）
        
        # 4. 重複チェック（同一画像パスは最新を優先）
        if img_path in unique_feedbacks:
            print(f"  ⚠ Duplicate image: {img_path} (keeping latest)")
            stats['duplicates'] += 1
            # 既存を上書き（最新を保持）
        
        unique_feedbacks[img_path] = fb
    
    # 有効なデータのみ残す
    cleaned_data = list(unique_feedbacks.values())
    stats['valid'] = len(cleaned_data)
    
    # 統計表示
    print("\n" + "=" * 60)
    print("Cleaning Results")
    print("=" * 60)
    print(f"Total:              {stats['total']}")
    print(f"Image not found:    {stats['image_not_found']}")
    print(f"Low coverage:       {stats['low_coverage']}")
    print(f"High coverage:      {stats['high_coverage']}")
    print(f"Low confidence:     {stats['low_confidence']} (not excluded)")
    print(f"Duplicates:         {stats['duplicates']}")
    print(f"Valid:              {stats['valid']}")
    print(f"Removal rate:       {(stats['total'] - stats['valid']) / stats['total'] * 100:.1f}%")
    
    # クラス別統計
    class_counts = defaultdict(int)
    for fb in cleaned_data:
        class_counts[fb['user_label']] += 1
    
    print("\nClass distribution (cleaned):")
    print(f"  sun:  {class_counts['sun']}")
    print(f"  moon: {class_counts['moon']}")
    print(f"  rain: {class_counts['rain']}")
    
    # クリーニング済みJSONを保存
    output_file = 'feedback_cleaned.json'
    with open(output_file, 'w') as f:
        json.dump({'feedbacks': cleaned_data}, f, indent=2)
    
    print(f"\nCleaned data saved to: {output_file}")
    print("=" * 60)
    
    return cleaned_data

if __name__ == '__main__':
    clean_feedback()
