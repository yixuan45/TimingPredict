import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
np.random.seed(42)


class BinanceLightGBM(object):
    def __init__(self):
        pass

    @staticmethod
    def get_data():
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
        return df1

    # 特征工程
    @staticmethod
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
            if i == 1: # 如果是当前时刻，就继续
                continue
            df_feat[f'lag_{i}'] = df_feat['c'].shift(i-1)

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
        delta = df_feat['c'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_feat['rsi'] = 100 - (100 / (1 + rs))

        # MACD指标
        df_feat['ema_12'] = df_feat['c'].ewm(span=12, adjust=False).mean()
        df_feat['ema_26'] = df_feat['c'].ewm(span=26, adjust=False).mean()
        df_feat['macd'] = df_feat['ema_12'] - df_feat['ema_26']
        df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
        df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']

        # 将当前时刻的c值，往前一段时间移动一位，即让未来时刻的数据作为当前时刻的预测目标
        df_feat['target']=df['c'].shift(-1)

        # 删除包含NaN的行
        df_feat = df_feat.dropna()
        df_feat = df_feat.reset_index(drop=True)  # 修正：添加drop=True避免保留原索引

        return df_feat

    # 划分数据集
    @staticmethod
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
        # 确保不包含日期列
        X = df.drop([target_col, 't'], axis=1) if 't' in df.columns else df.drop([target_col], axis=1)
        y = df[target_col]

        # 转换为NumPy数组
        X = X.values
        y = y.values

        # 划分训练集和测试集2025-01-01
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def standar(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # 训练LightGBM模型
    @staticmethod
    def train_lightgbm_model(train_data, test_data, X_test_scaled):
        """
        训练LightGBM模型并进行预测

        参数:
        train_data (lgb.Dataset): 训练数据集
        test_data (lgb.Dataset): 测试数据集
        X_test_scaled (numpy.ndarray): 标准化后的测试特征

        返回:
        tuple: 包含训练好的模型和预测结果的元组
        """
        # 设置模型参数
        params = {
            'objective': 'regression',  # 回归任务
            'metric': 'rmse',  # 评估指标为均方根误差
            'max_depth': 5,  # 树的最大深度
            'learning_rate': 0.1,  # 学习率
            'num_leaves': 31,  # 叶子节点数
            'feature_fraction': 0.8,  # 特征采样比例
            'bagging_fraction': 0.8,  # 样本采样比例
            'bagging_freq': 5,  # 样本采样频率
            'verbose': -1,  # 不输出信息
            'seed': 42  # 随机种子
        }

        # 添加早停机制
        early_stopping_rounds = 50

        # 训练模型
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[test_data],  # 只在验证集上评估
            valid_names=['validation']
        )

        # 使用标准化后的测试数据进行预测
        y_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)

        # 打印最佳迭代次数和验证集上的最佳分数
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score['validation']['rmse']}")

        return model, y_pred

    # 评估模型
    @staticmethod
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
        # 避免除零错误
        valid_indices = y_test != 0
        mape = np.mean(np.abs((y_test[valid_indices] - y_pred[valid_indices]) / y_test[valid_indices])) * 100

        # 创建评估指标字典
        metrics = {
            '均方误差(MSE)': mse,
            '均方根误差(RMSE)': rmse,
            '平均绝对误差(MAE)': mae,
            'R²分数(R2)': r2,
            '平均绝对百分比误差(MAPE)': mape
        }

        return metrics

    # 可视化特征重要性
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """
        可视化特征重要性

        参数:
        model (lgb.Booster): 训练好的LightGBM模型
        feature_names (list): 特征名称列表
        """
        lgb.plot_importance(model, importance_type='gain', figsize=(12, 8))
        plt.title('特征重要性')
        plt.tight_layout()
        plt.show()

        # 打印特征重要性分数
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("特征重要性:")
        print(feature_importance)

    @staticmethod
    # 可视化结果
    def visualize_results(df, y_test, y_pred, stock_code, train_size, dates):
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
        test_dates = dates[train_size+1:]
        predictions_df = pd.DataFrame({'Date': test_dates, '真实价格': y_test[:-1], '预测价格': y_pred[:-1]})
        predictions_df.set_index('Date', inplace=True)

        # 可视化预测结果
        plt.figure(figsize=(14, 7))

        # 绘制原始价格曲线
        plt.subplot(2, 1, 1)
        plt.plot(dates, df['c'], label='原始价格', color='blue')
        plt.plot(dates[train_size+1:], predictions_df['预测价格'], label='预测价格', color='green', linestyle='-.')
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

    def run(self):
        # 1.首先读取数据
        df = self.get_data()

        # 2.对数据进行特征提取
        df_feat = self.create_features(df=df)

        # 3.去除掉索引值为index的列
        df2=df_feat.copy()

        # 4.划分数据集
        dates = pd.to_datetime(df2['t'], unit='ms')
        X_train, X_test, y_train, y_test = self.split_data(df2, 'target', test_size=0.3)

        # 保存特征名称用于特征重要性分析
        feature_names = df2.drop(['target', 't'], axis=1).columns.tolist()

        # 5.对数据进行标准化
        X_train_scaled, X_test_scaled = self.standar(X_train, X_test)  # 返回标准化后的数据

        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

        # 6.训练LightGBM模型
        model, y_pred = self.train_lightgbm_model(train_data, test_data, X_test_scaled)  # 修正：传入标准化后的测试数据

        # 7.评估模型的效果
        metrics = self.evaluate_model(y_test, y_pred)
        print("模型评估指标:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 8.可视化特征重要性
        self.plot_feature_importance(model, feature_names)

        # 9.呈现预测可视化的结果
        train_size = len(y_train)
        self.visualize_results(df2, y_test, y_pred, 'BTCUSDT', train_size=train_size, dates=dates)


if __name__ == '__main__':
    lgbm = BinanceLightGBM()
    lgbm.run()