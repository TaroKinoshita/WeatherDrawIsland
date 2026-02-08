import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import glob
import json
from collections import defaultdict

# ========== 設定 ==========
FEEDBACK_JSON = 'feedback_cleaned.json'  # クリーニング済みJSON
UNITY_PROJECT_PATH = r'C:\UnityProjects\Weather Draw Islands\Assets\MyFolder' 
EXISTING_MODEL = 'weather_model_v6.pth'  # 既存モデル（Fine-tuning用）

# ========== 既存データセット（フォルダ構造） ==========
class WeatherDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        # クラスとラベルのマッピング
        self.class_to_idx = {'sun': 0, 'moon': 1, 'rain': 2}
        
        # 各クラスフォルダから画像を読み込み
        for class_name, label in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
                
            image_files = glob.glob(os.path.join(class_path, '*.png'))
            
            for img_path in image_files:
                self.data.append(img_path)
                self.labels.append(label)
        
        print(f"Loaded {len(self.data)} images from {data_dir}")
        print(f"  sun: {self.labels.count(0)}, moon: {self.labels.count(1)}, rain: {self.labels.count(2)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # 画像読み込み（グレースケール）
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ========== フィードバックデータセット（JSON） ==========
class FeedbackDataset(Dataset):
    def __init__(self, json_path, unity_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        # クラスマッピング
        self.class_to_idx = {'sun': 0, 'moon': 1, 'rain': 2}
        
        # JSON読み込み
        with open(json_path, 'r') as f:
            feedback_data = json.load(f)
        
        feedbacks = feedback_data['feedbacks']
        
        for fb in feedbacks:
            img_path = os.path.join(unity_path, fb['image_path'])
            user_label = fb['user_label']
            
            if user_label not in self.class_to_idx:
                print(f"Warning: Unknown label '{user_label}' in {img_path}")
                continue
            
            self.data.append(img_path)
            self.labels.append(self.class_to_idx[user_label])
        
        print(f"Loaded {len(self.data)} images from feedback")
        print(f"  sun: {self.labels.count(0)}, moon: {self.labels.count(1)}, rain: {self.labels.count(2)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # 画像読み込み（グレースケール）
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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

# ========== 再学習（Fine-tuning） ==========
def retrain_model():
    print("=" * 60)
    print("Retraining with Feedback Data")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # データ拡張（既存と同じ）
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 検証用（拡張なし）
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 既存データセット読み込み
    existing_dataset = None
    if os.path.exists('train'):
        existing_dataset = WeatherDataset('train', transform=transform)
    else:
        print("Warning: 'train' folder not found. Using only feedback data.")
    
    # フィードバックデータセット読み込み
    feedback_dataset = None
    if os.path.exists(FEEDBACK_JSON):
        feedback_dataset = FeedbackDataset(FEEDBACK_JSON, UNITY_PROJECT_PATH, transform=transform)
    else:
        print(f"Error: {FEEDBACK_JSON} not found!")
        return
    
    # データセット統合
    if existing_dataset is not None and feedback_dataset is not None:
        combined_dataset = ConcatDataset([existing_dataset, feedback_dataset])
        print(f"\nCombined dataset size: {len(combined_dataset)}")
    elif feedback_dataset is not None:
        combined_dataset = feedback_dataset
        print(f"\nUsing only feedback data: {len(combined_dataset)}")
    else:
        print("Error: No data available!")
        return
    
    # 訓練/検証分割（80%/20%）
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train size: {train_size}, Val size: {val_size}\n")
    
    # モデル読み込み（Fine-tuning）
    model = SimpleCNN().to(device)
    
    if os.path.exists(EXISTING_MODEL):
        print(f"Loading existing model: {EXISTING_MODEL}")
        model.load_state_dict(torch.load(EXISTING_MODEL, map_location=device))
        print("Model loaded successfully\n")
    else:
        print(f"Warning: {EXISTING_MODEL} not found. Training from scratch.\n")
    
    # 損失関数・最適化手法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Fine-tuning用に学習率を下げる
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5,  # Fine-tuning用にpatienceを短く
        verbose=True
    )
    
    # Early Stopping
    patience = 15
    best_acc = 0.0
    no_improve_epochs = 0
    
    # 学習ループ
    num_epochs = 50  # Fine-tuningなので少なめ
    
    print("=" * 60)
    print("Training Start")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # 検証モード
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # 学習率スケジューラ更新
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")
        
        # ベストモデル保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'weather_model_retrained.pth')
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early Stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
            break
    
    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2f}%")
    
    # ONNXエクスポート
    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load('weather_model_retrained.pth'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'weather_model_retrained.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("ONNX model exported: weather_model_retrained.onnx")
    print("=" * 60)

    # ONNXエクスポート後
    print("ONNX model exported: weather_model_retrained.onnx")

    # 自動コピー
    import shutil
    unity_model_path = 'C:/UnityProjects/Weather Draw Islands/Assets/MyFolder/Models/weather_model_retrained.onnx'
    shutil.copy('weather_model_retrained.onnx', unity_model_path)
    print(f"ONNX copied to Unity: {unity_model_path}")

if __name__ == '__main__':
    retrain_model()
