import numpy as np
from PIL import Image
import os

# --- 設定セクション ---
DATA_DIR = r'C:\Users\kinos\OneDrive\デスクトップ\MyProject\NewMediaProject\Python_ONNX\data'
TRAIN_DIR = r'C:\Users\kinos\OneDrive\デスクトップ\MyProject\NewMediaProject\Python_ONNX\train'

file_map = {
    'sun.npy': 'sun',
    'moon.npy': 'moon',
    'cloud.npy': 'rain'
}

def export_samples_original(count=50):
    for npy_file, folder_name in file_map.items():
        npy_path = os.path.join(DATA_DIR, npy_file)
        target_dir = os.path.join(TRAIN_DIR, folder_name)

        if not os.path.exists(npy_path):
            print(f"❌ Skip: {npy_file} が {DATA_DIR} にねーぞ。")
            continue
        
        # .npyをロード
        data = np.load(npy_path)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(f"🔄 Processing {folder_name} (28x28 original)...")
        for i in range(min(count, len(data))):
            # 28x28にリシェイプ
            img_array = data[i].reshape(28, 28)
            # そのまま画像化
            img = Image.fromarray(img_array.astype('uint8'))
            
            # リサイズせずにそのまま保存
            save_path = os.path.join(target_dir, f"{folder_name}_{i:03d}.png")
            img.save(save_path)
            
    print("\n✅ 28x28のオリジナルサイズで書き出し完了。")

if __name__ == "__main__":
    export_samples_original()