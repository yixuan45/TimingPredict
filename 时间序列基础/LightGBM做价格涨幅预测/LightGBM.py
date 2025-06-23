import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
np.random.seed(42)


class LightGbmIncrease(object):
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
            if i == 1:  # 如果是当前时刻，就继续
                continue
            df_feat[f'lag_{i}'] = df_feat['c'].shift(i - 1)

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
        df_feat['diff'] = df_feat['c'].shift(-1) - df_feat['c']
        df_feat.loc[df_feat['diff'] > 0, 'target'] = 1  # 上涨标记为1
        df_feat.loc[df_feat['diff'] <= 0, 'target'] = -1  # 下跌或持平标记为-1

        df_feat = df_feat.drop('diff', axis=1)
        # 删除包含NaN的行
        df_feat = df_feat.dropna()
        df_feat = df_feat.reset_index(drop=True)  # 修正：添加drop=True避免保留原索引

        return df_feat

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

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_lightgbm_model(train_data, test_data, X_test_scaled, y_test):
        # 设置参数，使用multiclass目标并设置num_class=2
        params = {
            'objective': 'multiclass',
            'num_class': 2,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42,
            'is_unbalance': True  # 金融数据通常存在类别不平衡
        }

        # 训练模型
        num_round = 100
        model = lgb.train(
            params,
            train_data,
            num_round,
            valid_sets=[test_data]
        )

        # 预测
        y_pred_proba = model.predict(X_test_scaled)
        # 获取预测类别索引
        y_pred_index = np.argmax(y_pred_proba, axis=1)
        # 将类别索引映射回-1和1
        y_pred = np.where(y_pred_index == 0, -1, 1)

        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        # 计算精确率、召回率和F1分数时，指定average='binary'和pos_label=1
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)

        # 计算ROC AUC，需要将-1/1转换为0/1
        y_test_binary = np.where(y_test == 1, 1, 0)
        # 取类别1的概率作为正类概率
        y_pred_proba_pos = y_pred_proba[:, 1]
        roc_auc = roc_auc_score(y_test_binary, y_pred_proba_pos)

        # 打印主要评估指标
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率 (上涨): {precision:.4f}")
        print(f"召回率 (上涨): {recall:.4f}")
        print(f"F1分数 (上涨): {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["预测下跌", "预测上涨"],
                    yticklabels=["实际下跌", "实际上涨"])
        plt.title("混淆矩阵")
        plt.xlabel("预测标签")
        plt.ylabel("实际标签")
        plt.savefig('confusion_matrix.png')
        plt.close()

        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=["下跌", "上涨"]))

        # 特征重要性
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(model, importance_type='gain', title='特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_pos)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征曲线')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

        # 精确率-召回率曲线
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_pred_proba_pos)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label='精确率-召回率曲线')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="upper right")
        plt.savefig('precision_recall_curve.png')
        plt.close()

        return model

    @staticmethod
    def standar(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def run(self):
        # 1.首先读取数据
        df = self.get_data()

        # 2.对数据进行特征提取
        df_feat = self.create_features(df=df)
        df2 = df_feat.copy()

        # 3.划分数据集
        dates = pd.to_datetime(df2['t'], unit='ms')  # 获得时间序列
        X_train, X_test, y_train, y_test = self.split_data(df2, 'target', test_size=0.3)

        # 4.保存特征名称用于特征重要性分析
        feature_names = df2.drop(['target', 't'], axis=1).columns.tolist()

        # 5.对数据进行标准化
        X_train_scaled, X_test_scaled = self.standar(X_train, X_test)  # 返回标准化后的数据

        # 创建LightGBM数据集
        # 将-1/1标签映射为0/1，因为LightGBM需要从0开始的类别索引
        y_train_mapped = np.where(y_train == 1, 1, 0)
        y_test_mapped = np.where(y_test == 1, 1, 0)

        train_data = lgb.Dataset(X_train_scaled, label=y_train_mapped)
        test_data = lgb.Dataset(X_test_scaled, label=y_test_mapped, reference=train_data)

        # 6.训练LightGBM模型
        model = self.train_lightgbm_model(train_data, test_data, X_test_scaled, y_test)




if __name__ == '__main__':
    lbgm = LightGbmIncrease()
    lbgm.run()