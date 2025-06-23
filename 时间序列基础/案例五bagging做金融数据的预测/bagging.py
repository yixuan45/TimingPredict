import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import os
from tqdm import tqdm

# 设置随机种子以确保结果可复现
np.random.seed(42)


def fetch_stock_data(ticker, start_date, end_date):
    """获取股票数据"""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"获取股票数据时出错: {e}")
        return None


def prepare_data(data, window_size=5, forecast_days=1):
    """准备训练数据和标签"""
    df = data.copy()
    df['Return'] = df['c'].pct_change()
    df = df.dropna()

    # 创建技术指标
    df['SMA_5'] = df['c'].rolling(window=5).mean()
    df['SMA_10'] = df['c'].rolling(window=10).mean()
    df['SMA_20'] = df['c'].rolling(window=20).mean()

    df['EMA_5'] = df['c'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['c'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['c'].ewm(span=20, adjust=False).mean()

    df['MACD'] = df['EMA_10'] - df['EMA_20']

    df['RSI'] = calculate_rsi(df['c'], period=14)

    df = df.dropna()

    # 创建特征和标签
    X = []
    y = []

    close_prices = df['c'].values
    features = df.drop(['c'], axis=1).values

    for i in range(len(df) - window_size - forecast_days + 1):
        X.append(features[i:i + window_size].flatten())
        y.append(close_prices[i + window_size + forecast_days - 1])

    return np.array(X), np.array(y), df.index[window_size + forecast_days - 1:]


def calculate_rsi(prices, period=14):
    """计算相对强弱指数(RSI)"""
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def train_bagging_model(X, y):
    """训练Bagging模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建和训练Bagging模型
    base_estimator = DecisionTreeRegressor(max_depth=5)
    bagging_model = BaggingRegressor(
        base_estimator,
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=True,
        n_jobs=-1,
        random_state=42
    )

    bagging_model.fit(X_train_scaled, y_train)

    # 在测试集上进行预测
    y_pred = bagging_model.predict(X_test_scaled)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")

    return bagging_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred


def plot_predictions(actual_prices, predicted_prices, dates, train_size):
    """可视化预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:train_size], actual_prices[:train_size], label='训练数据', color='blue')
    plt.plot(dates[train_size:], actual_prices[train_size:], label='实际价格', color='green')
    plt.plot(dates[train_size:], predicted_prices, label='预测价格', color='red', linestyle='--')
    plt.title('股票价格预测与实际走势对比')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def predict_future_prices(model, scaler, recent_data, window_size, forecast_days=5):
    """预测未来价格"""
    last_date = recent_data.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    # 准备预测数据
    X_pred = []
    features = recent_data.drop(['c'], axis=1).values

    # 使用最近的window_size天数据进行预测
    X_pred.append(features[-window_size:].flatten())
    X_pred_scaled = scaler.transform(X_pred)

    # 预测未来价格
    future_prices = []
    current_features = X_pred_scaled[0].copy()

    for _ in range(forecast_days):
        # 预测下一天价格
        next_price = model.predict([current_features])[0]
        future_prices.append(next_price)

        # 更新特征向量，移除最早的一天数据，添加新预测的价格对应的特征
        # 这里简化处理，实际应用中应重新计算所有技术指标
        current_features = np.roll(current_features, -len(recent_data.columns) + 2)
        # 添加新的价格相关特征（简化处理）
        current_features[-len(recent_data.columns) + 2] = next_price

    return future_dates, future_prices


def main():
    """主函数"""
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

    window_size = 10  # 用于预测的历史天数
    forecast_days = 1  # 预测未来的天数

    # 准备数据
    X, y, dates = prepare_data(df1, window_size, forecast_days)

    # 训练模型
    model, scaler, X_train, X_test, y_train, y_test, y_pred = train_bagging_model(X, y)

    # 可视化预测结果
    train_size = len(y_train)
    actual_prices = np.concatenate([y_train, y_test])
    plot_predictions(actual_prices, y_pred, dates, train_size)

if __name__ == "__main__":
    main()