import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 数据读取与增强
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 修正可能的列名拼写错误
    if 'temparature' in train.columns:
        train.rename(columns={'temparature': 'temperature'}, inplace=True)
        test.rename(columns={'temparature': 'temperature'}, inplace=True)
    
    # 扩展特征工程
    for df in [train, test]:
        # 基础特征
        df['temp_diff'] = df['maxtemp'] - df['mintemp']
        df['dew_temp_ratio'] = df['dewpoint'] / (df['temperature'] + 1e-6)
        
        # 新增交互特征
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['pressure_windspeed'] = df['pressure'] * df['windspeed']
        
        # 分箱特征
        df['temp_bin'] = pd.cut(df['temperature'], 5, labels=False)
        df['humidity_bin'] = pd.qcut(df['humidity'], 4, labels=False)
        
        # 时间特征（假设数据包含时间字段）
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['season'] = df['month'] % 12 // 3 + 1
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    cols = [c for c in train.columns if c not in ['id', 'rainfall', 'date']]
    train[cols] = imputer.fit_transform(train[cols])
    test[cols] = imputer.transform(test[cols])
    
    return train, test

# 优化后的模型训练
def train_model(X_train, y_train):
    model = LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=3000,
        learning_rate=0.0025,
        max_depth=-1,  # 使用树模型默认深度
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        device='gpu' 
    )
    model.fit(X_train, y_train)
    return model

# 主流程优化
def main():
    train_path = r'D:\C_data\kaggle\降雨量\train.csv'
    test_path = r'D:\C_data\kaggle\降雨量\test.csv'
    
    train, test = load_data(train_path, test_path)
    
    # 更新特征列表
    features = ['pressure', 'maxtemp', 'mintemp', 'dewpoint', 
               'humidity', 'cloud', 'sunshine', 'windspeed',
               'temp_diff', 'dew_temp_ratio', 'temp_humidity',
               'pressure_windspeed', 'temp_bin', 'humidity_bin']
    
    if 'month' in train.columns:
        features += ['month', 'season']
    
    X = train[features]
    y = train['rainfall']
    
    # 交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = train_model(X_train, y_train)
        oof_preds[valid_idx] = model.predict_proba(X_val)[:, 1]
        
        # 早停机制
        val_score = roc_auc_score(y_val, oof_preds[valid_idx])
        print(f'Fold {fold+1} ROC AUC: {val_score:.4f}')
    
    print(f'Overall ROC AUC: {roc_auc_score(y, oof_preds):.4f}')
    
    # 全量训练
    final_model = train_model(X, y)
    
    # 测试集预测
    test_probs = final_model.predict_proba(test[features])[:, 1]
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test['id'],
        'rain_prob': test_probs
    })
    submission.to_csv(r'D:\C_data\kaggle\降雨量\submission.csv', index=False)
    print('Submission file generated.')

if __name__ == "__main__":
    main()
