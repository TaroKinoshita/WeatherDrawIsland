import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

# ========== データセット定義 ==========
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
            image_files = glob.glob(os.path.join(class_path, '*.png'))
            
            for img_path in image_files:
                self.data.append(img_path)
                self.labels.append(label)
        
        print(f"Loaded {len(self.data)} images")
        print(f"sun: {self.labels.count(0)}, moon: {self.labels.count(1)}, rain: {self.labels.count(2)}")
    
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

# ========== CNNモデル定義 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 7x7 -> 7x7
        
        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 3)  # 3クラス分類
        
        # 活性化関数・ドロップアウト
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        x = x.view(-1, 64 * 3 * 3)  # 平坦化
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# ========== 学習ループ ==========
def train_model():
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データ前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])
    
    # データセット読み込み
    dataset = WeatherDataset('train', transform=transform)
    
    # 訓練/検証データに分割（80%/20%）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # モデル・損失関数・最適化手法
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学習
    num_epochs = 30
    best_acc = 0.0
    
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")
        
        # ベストモデル保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'drawing_model.pth')
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2f}%")
    
    # ONNXエクスポート
    model.load_state_dict(torch.load('drawing_model.pth'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'weather_model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("ONNX model exported: weather_model.onnx")

if __name__ == '__main__':
    train_model()