import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CNNモデル（既存と同じ） ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 128 * 3 * 3)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ========== データセット（評価用） ==========
class WeatherDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        self.class_to_idx = {'sun': 0, 'moon': 1, 'rain': 2}
        
        for class_name, label in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            image_files = glob.glob(os.path.join(class_path, '*.png'))
            
            for img_path in image_files:
                self.data.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ========== モデル評価 ==========
def evaluate_model(model_path, test_dir='test', output_prefix=''):
    """
    モデルを評価し、混同行列とレポートを出力
    
    Args:
        model_path: モデルファイルパス (.pth)
        test_dir: テストデータディレクトリ
        output_prefix: 出力ファイルのプレフィックス（before/after識別用）
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}\n")
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データ前処理（評価時は拡張なし）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # テストデータ読み込み
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory '{test_dir}' not found.")
        print("Using validation split from train directory instead.")
        test_dir = 'train'
    
    test_dataset = WeatherDataset(test_dir, transform=transform)
    
    if len(test_dataset) == 0:
        print("Error: No test data found!")
        return
    
    print(f"Test dataset size: {len(test_dataset)}\n")
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # モデル読み込み
    model = SimpleCNN().to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded: {model_path}\n")
    
    # 予測
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 精度計算
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Accuracy: {accuracy:.2f}%\n")
    
    # クラス名
    class_names = ['sun', 'moon', 'rain']
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # 分類レポート
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 混同行列を画像として保存
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {output_prefix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_file = f'confusion_matrix_{output_prefix}.png' if output_prefix else 'confusion_matrix.png'
    plt.savefig(output_file)
    print(f"Confusion matrix saved: {output_file}")
    
    plt.close()
    
    return accuracy, cm

# ========== Before/After比較 ==========
def compare_models():
    """
    再学習前後のモデルを比較
    """
    print("\n" + "=" * 60)
    print("Model Comparison: Before vs After Retraining")
    print("=" * 60)
    
    results = {}
    
    # Before（既存モデル）
    if os.path.exists('weather_model_v6.pth'):
        print("\n[BEFORE] Evaluating original model...")
        acc_before, cm_before = evaluate_model('weather_model_v6.pth', output_prefix='before')
        results['before'] = {'accuracy': acc_before, 'confusion_matrix': cm_before}
    else:
        print("\nWarning: Original model (weather_model_v6.pth) not found.")
        results['before'] = None
    
    # After（再学習モデル）
    if os.path.exists('weather_model_retrained.pth'):
        print("\n[AFTER] Evaluating retrained model...")
        acc_after, cm_after = evaluate_model('weather_model_retrained.pth', output_prefix='after')
        results['after'] = {'accuracy': acc_after, 'confusion_matrix': cm_after}
    else:
        print("\nWarning: Retrained model (weather_model_retrained.pth) not found.")
        results['after'] = None
    
    # 比較レポート
    if results['before'] and results['after']:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"BEFORE: {results['before']['accuracy']:.2f}%")
        print(f"AFTER:  {results['after']['accuracy']:.2f}%")
        
        diff = results['after']['accuracy'] - results['before']['accuracy']
        if diff > 0:
            print(f"Improvement: +{diff:.2f}% ✓")
        elif diff < 0:
            print(f"Degradation: {diff:.2f}% ✗")
        else:
            print("No change")
        print("=" * 60)
    
    return results

if __name__ == '__main__':
    # 引数があれば単体評価、なければ比較評価
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        evaluate_model(model_path)
    else:
        compare_models()
