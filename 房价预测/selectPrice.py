# -*- coding: utf-8 -*-
"""
房价预测深度学习流程
包含数据预处理、特征工程、神经网络建模和预测输出完整流程
"""

# 基础库导入
import pandas as pd
import numpy as np

# 机器学习预处理工具
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 深度学习框架
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------- 数据读取阶段 -----------------
# 读取训练集和测试集数据，使用原始数据路径
train = pd.read_csv(r'D:\C_data\房价预测2\train.csv')
test = pd.read_csv(r'D:\C_data\房价预测2\test.csv')

# ----------------- 数据预处理阶段 -----------------
# 1. 缺失值处理：删除缺失值超过800的列
missing_counts = train.isnull().sum()  # 计算每列缺失值数量
cols_to_drop = missing_counts[missing_counts > 800].index.tolist()  # 确定要删除的列
train = train.drop(cols_to_drop, axis=1)  # 从训练集删除
test = test.drop(cols_to_drop, axis=1)    # 同步删除测试集对应列

# 2. 特征相关性处理（基于训练集）
# 筛选数值型特征（排除ID和目标变量）
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]

# 计算绝对值相关系数矩阵
corr_matrix = train[numeric_cols].corr().abs()
# 取上三角矩阵（避免重复比较）
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# 标记相关系数>0.8的特征
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# 同步删除高相关特征
train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)

# ----------------- 特征工程阶段 -----------------
# 分离目标变量和ID列
y_train = train['SalePrice']        # 训练集目标变量
test_ids = test['Id']               # 保留测试集ID用于最终输出
train = train.drop(['SalePrice'], axis=1)  # 移除训练集目标变量

# 构建预处理管道
# 定义数值型特征处理流程（需要先移除ID列）
numeric_features = train.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('Id')  # 移除ID列

# 定义分类型特征
categorical_features = train.select_dtypes(exclude=np.number).columns.tolist()

# 数值型处理：缺失值填充->标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 均值填充
    ('scaler', StandardScaler())])                # 标准化处理

# 分类型处理：缺失值填充->独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # 处理未知类别

# 组合预处理流程
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),    # 数值处理通道
        ('cat', categorical_transformer, categorical_features)],  # 分类处理通道
    remainder='drop')  # 丢弃未指定的列

# 执行预处理（保持ID列不参与处理）
X_train = preprocessor.fit_transform(train.drop('Id', axis=1))  # 训练集预处理
X_test = preprocessor.transform(test.drop('Id', axis=1))        # 测试集预处理

# ----------------- 模型构建阶段 -----------------
# 创建深度神经网络模型
model = Sequential([
    # 输入层：128个神经元，ReLU激活，输入维度根据特征数量自动适配
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    # 隐藏层：64个神经元
    Dense(64, activation='relu'),
    # 隐藏层：32个神经元
    Dense(32, activation='relu'),
    # 输出层：1个神经元（回归问题无激活函数）
    Dense(1)
])

# 模型编译配置
model.compile(
    optimizer='adam',  # 自适应矩估计优化器
    loss='mse'         # 使用均方误差作为损失函数
)

# ----------------- 模型训练阶段 -----------------
# 训练模型（保留20%作为验证集）
history = model.fit(
    X_train, y_train,
    epochs=100,        # 训练轮数
    batch_size=32,     # 批量大小
    validation_split=0.2  # 验证集比例
)

# ----------------- 预测输出阶段 -----------------
# 生成测试集预测结果
predictions = model.predict(X_test)

# 创建结果DataFrame并保存
output = pd.DataFrame({
    'Id': test_ids,                # 测试集ID
    'SalePrice': predictions.flatten()  # 展平预测结果
})
output.to_csv('predictions2.csv', index=False)  # 输出CSV文件
