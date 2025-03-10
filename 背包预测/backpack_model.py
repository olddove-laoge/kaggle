import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 数据预处理（保持不变）
def preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                       'Waterproof', 'Style', 'Color']
    numerical_cols = ['Compartments', 'Weight Capacity (kg)']

    for df in [train_df, test_df]:
        df[categorical_cols] = df[categorical_cols].fillna('Missing')
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ])

    # 训练集处理
    X_train = preprocessor.fit_transform(train_df.drop(['id', 'Price'], axis=1))
    y_train = train_df['Price'].values
    
    # 测试集处理
    X_test = preprocessor.transform(test_df.drop(['id'], axis=1))
    
    return X_train, y_train, X_test, test_df['id']

# PyTorch模型定义
class BackpackModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # 数据预处理
    X_train, y_train, X_test, test_ids = preprocess_data(r'D:\C_data\backpack预测\train.csv', r'D:\C_data\backpack预测\test.csv')
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train.toarray())  # 处理稀疏矩阵
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.toarray())
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 将数据移动到设备上
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 初始化模型
    model = BackpackModel(X_train.shape[1]).to(device)  # 将模型移动到设备上
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # 训练参数
    epochs = 200
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # 确保数据在设备上
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_pred = model(X_train_tensor)
            val_loss = criterion(val_pred, y_train_tensor)
        
        # 早停机制
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 生成预测
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()  # 将预测结果移动到CPU上
    
    # 创建结果文件
    result_df = pd.DataFrame({
        'id': test_ids,
        'Price': np.clip(predictions, a_min=0, a_max=None)
    })
    
    result_df.to_csv(r'D:\C_data\backpack预测\predictions.csv', index=False)
    print("预测结果已保存至 predictions.csv")
