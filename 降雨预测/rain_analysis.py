import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

# 配置GPU加速
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocess_data(train_paths, test_path):
    # 读取并合并多个训练文件
    train_dfs = [pd.read_csv(path) for path in train_paths]
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    test_df = pd.read_csv(test_path)
    
    # 定义特征列
    categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    numerical_cols = ['Compartments', 'Weight Capacity (kg)']
    
    # 计算训练集的统计量
    num_means = train_df[numerical_cols].mean()
    
    # 处理缺失值（使用训练集统计量填充测试集）
    train_df[categorical_cols] = train_df[categorical_cols].fillna('Missing')
    train_df[numerical_cols] = train_df[numerical_cols].fillna(num_means)
    
    test_df[categorical_cols] = test_df[categorical_cols].fillna('Missing')
    test_df[numerical_cols] = test_df[numerical_cols].fillna(num_means)
    
    # 构建预处理管道
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])
    
    # 处理训练数据
    X_train = preprocessor.fit_transform(train_df.drop(['id', 'Price'], axis=1))
    y_train = np.log1p(train_df['Price'].values)  # 对数变换处理价格偏态
    
    # 处理测试数据
    X_test = preprocessor.transform(test_df.drop('id', axis=1))
    
    return X_train, y_train, X_test, test_df['id']

def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='leaky_relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='leaky_relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1)
    ])
    
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber_loss')  # 使用Huber损失增强鲁棒性
    return model

if __name__ == "__main__":
    train_files = [
        r'D:\C_data\backpack预测\train.csv',
        r'D:\C_data\backpack预测\training_extra.csv',
    ]
    test_file = r'D:\C_data\backpack预测\test.csv'
    
    # 数据预处理
    X_train, y_train, X_test, test_ids = preprocess_data(train_files, test_file)
    
    # 创建模型
    model = create_model(X_train.shape[1])
    
    # 回调函数配置
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=64,  # 增大批大小适应GPU并行
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # 加载最佳模型
    model.load_weights('best_model.h5')
    
    # 生成预测并还原对数变换
    predictions = np.expm1(model.predict(X_test).flatten())
    
    # 保存结果
    pd.DataFrame({
        'id': test_ids,
        'Price': np.clip(predictions, a_min=0, a_max=None)
    }).to_csv(r'D:\C_data\backpack预测\predictions.csv', index=False)
    
    print("预测完成，结果已保存至 predictions.csv")
