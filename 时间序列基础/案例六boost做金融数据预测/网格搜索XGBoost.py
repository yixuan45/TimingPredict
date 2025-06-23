import os
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 获取数据
np.random.seed(42)
# 读取数据并处理数据
path = "Z:\data\BTCUSDT\BTCUSDT-1h"
filelist = os.listdir(path)
filelist.sort()
df = pd.DataFrame()
for i in tqdm(filelist[-300:]):
    df0 = pd.read_csv(os.path.join(path, i))
    df = pd.concat([df, df0])
df1 = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
df1.columns = ['t', 'o', 'h', 'l', 'c', 'v']
df1 = df1.dropna()

# 特征工程
def create_features(df):
    """
    创建用于预测的特征

    参数:
    df (pandas.DataFrame): 包含股票数据的DataFrame

    返回:
    pandas.DataFrame: 包含特征的DataFrame
    """
    df_feat = df.copy()

    # 创建滞后特征
    for i in range(1, 6):  # 1到5天的滞后
        df_feat[f'lag_{i}'] = df_feat['c'].shift(i)

    # 创建移动平均特征
    df_feat['MA_5'] = df_feat['c'].rolling(window=5).mean()
    df_feat['MA_10'] = df_feat['c'].rolling(window=10).mean()
    df_feat['MA_20'] = df_feat['c'].rolling(window=20).mean()

    # 创建波动率特征
    df_feat['Volatility_5'] = df_feat['c'].rolling(window=5).std()
    df_feat['Volatility_10'] = df_feat['c'].rolling(window=10).std()

    # 创建涨跌特征
    df_feat['Return'] = df_feat['c'].pct_change()
    df_feat['Return_prev'] = df_feat['Return'].shift(1)
    df_feat['Return_prev_2'] = df_feat['Return'].shift(2)

    # 创建价格变动特征
    df_feat['Price_change'] = df_feat['c'] - df_feat['o']
    df_feat['High_low'] = df_feat['h'] - df_feat['l']

    # RSI指标
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD指标
    df['ema_12'] = df['c'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 删除包含NaN的行
    df_feat = df_feat.dropna()
    df_feat = df_feat.reset_index()

    return df_feat

df2=create_features(df1)
df2=df2.drop(columns='index')

# 划分数据集
def split_data(df, target_col, test_size=0.2):
    """
    划分数据集为训练集和测试集

    参数:
    df (pandas.DataFrame): 包含特征和目标变量的DataFrame
    target_col (str): 目标变量列名
    test_size (float): 测试集比例，默认为0.2

    返回:
    tuple: 包含训练集特征、测试集特征、训练集目标、测试集目标的元组
    """
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test

dates = pd.to_datetime(df2['t'], unit='ms')
X_train, X_test, y_train, y_test = split_data(df2, 'c',test_size=0.3)

# 数据标准化
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# 网格搜索
# 网格搜索优化XGBoost模型参数
def grid_search_xgboost(X_train, y_train):
    """
    使用网格搜索找到XGBoost模型的最优参数

    参数:
    X_train (pandas.DataFrame): 训练集特征
    y_train (pandas.Series): 训练集目标

    返回:
    tuple: 包含最优模型和最优参数的元组
    """
    # 定义要搜索的参数网格
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # 创建XGBoost回归模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # 使用负均方误差作为评分标准
        cv=3,  # 3折交叉验证
        n_jobs=-1,  # 使用所有CPU核心
        verbose=2
    )

    # 执行网格搜索
    print("开始执行网格搜索...")
    grid_search.fit(X_train, y_train)

    # 打印最优参数
    print("最优参数:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    # 返回最优模型
    return grid_search.best_estimator_, grid_search.best_params_

# 网格搜索最优参数
best_model, best_params = grid_search_xgboost(X_train_scaled, y_train)

# 使用最优参数训练XGBoost模型
def train_xgboost_model(X_train, y_train, X_test, best_params):
    """
    使用最优参数训练XGBoost回归模型并进行预测

    参数:
    X_train (pandas.DataFrame): 训练集特征
    y_train (pandas.Series): 训练集目标
    X_test (pandas.DataFrame): 测试集特征
    best_params (dict): 最优参数

    返回:
    tuple: 包含训练好的模型和预测结果的元组
    """
    # 创建DMatrix格式的数据
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # 设置模型参数，使用网格搜索得到的最优参数
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'n_estimators': best_params['n_estimators'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'seed': 42
    }

    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])

    # 预测
    y_pred = model.predict(dtest)

    return model, y_pred

model, y_pred = train_xgboost_model(X_train_scaled, y_train, X_test_scaled, best_params)

# 评估模型
def evaluate_model(y_test, y_pred):
    """
    评估模型性能并返回各种评估指标

    参数:
    y_test (pandas.Series): 真实值
    y_pred (numpy.ndarray): 预测值

    返回:
    dict: 包含各种评估指标的字典
    """
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 计算平均绝对百分比误差(MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 创建评估指标字典
    metrics = {
        '均方误差(MSE)': mse,
        '均方根误差(RMSE)': rmse,
        '平均绝对误差(MAE)': mae,
        'R²分数(R2)': r2,
        '平均绝对百分比误差(MAPE)': mape
    }

    return metrics

metrics=evaluate_model(y_test, y_pred)
print(metrics)

# 可视化结果
def visualize_results(df, y_test, y_pred, stock_code,train_size,dates):
    """
    可视化预测结果和真实结果

    参数:
    df (pandas.DataFrame): 原始数据集
    y_test (pandas.Series): 真实值
    y_pred (numpy.ndarray): 预测值
    stock_code (str): 股票代码
    train_size: 训练集的大小
    dates: 时间列表
    """
    # 创建预测结果DataFrame
    test_dates = y_test.index
    predictions_df = pd.DataFrame({'Date': test_dates, '真实价格': y_test, '预测价格': y_pred})
    predictions_df.set_index('Date', inplace=True)

    # 可视化预测结果
    plt.figure(figsize=(14, 7))

    # 绘制原始价格曲线
    plt.subplot(2, 1, 1)
    plt.plot(dates, df['c'], label='原始价格', color='blue')
    plt.plot(dates[train_size:],predictions_df['预测价格'], label='预测价格', color='green',linestyle='-.')
    plt.title(f'{stock_code}股票原始价格')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)


    # 绘制预测价格和真实价格对比
    plt.subplot(2, 1, 2)
    plt.plot(predictions_df.index, predictions_df['真实价格'], label='真实价格', color='blue')
    plt.plot(predictions_df.index, predictions_df['预测价格'], label='预测价格', color='red', linestyle='--')
    plt.title(f'{stock_code}股票价格预测结果')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制预测误差
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['真实价格'] - predictions_df['预测价格'],
             label='预测误差', color='purple')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title(f'{stock_code}股票价格预测误差')
    plt.xlabel('日期')
    plt.ylabel('误差')
    plt.legend()
    plt.grid(True)
    plt.show()

train_size = len(y_train)
visualize_results(df2,y_test,y_pred,'BTCUSDT',train_size=train_size,dates=dates)