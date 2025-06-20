import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

data = pd.read_csv('./weather_classification_data.csv')
data.head()

data1 = data[
    ['Temperature', 'Humidity', 'Wind Speed', 'Cloud Cover', 'Atmospheric Pressure', 'Season', 'Visibility (km)',
     'Location', 'Weather Type']].copy()


# 定义颜色方案
colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F']

plt.figure(figsize=(16, 12))
gs = GridSpec(ncols=2, nrows=3, figure=plt.gcf())

# 绘制1温度曲线图
ax1 = plt.subplot(gs[0, 0])
sns.lineplot(data1['Temperature'][-200:], color=colors[0], ax=ax1)
ax1.set_title('温度曲线图')
ax1.set_ylabel('温度 (℃)')

# 绘制2湿度曲线图
ax2 = plt.subplot(gs[0, 1])
sns.lineplot(data1['Humidity'][-200:], color=colors[1], ax=ax2)
ax2.set_title('湿度曲线图')
ax2.set_ylabel('湿度')

# 绘制3风速曲线图
ax3 = plt.subplot(gs[1, 0])
sns.lineplot(data1['Wind Speed'][-200:], color=colors[2], ax=ax3)
ax3.set_title('风速曲线图')
ax3.set_ylabel('风速')

# 绘制4大气压曲线图
ax4 = plt.subplot(gs[1, 1])
sns.lineplot(data1['Atmospheric Pressure'][-200:], color=colors[3], ax=ax4)
ax4.set_title('大气压曲线图')
ax4.set_ylabel('大气压')

# 绘制5能见度曲线图
ax5 = plt.subplot(gs[2, 0])
sns.lineplot(data1['Visibility (km)'][-200:], color=colors[4], ax=ax5)
ax5.set_title('能见度曲线图')
ax5.set_ylabel('能见度')


# 定义颜色方案
colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F']

plt.figure(figsize=(16, 12))
gs = GridSpec(ncols=2, nrows=2, figure=plt.gcf())

# 1. Cloud Cover直方图
ax1 = plt.subplot(gs[0, 0])
sns.histplot(data1['Cloud Cover'], color=colors[0], ax=ax1)
ax1.set_title('Cloud Cover分布直方图')
ax1.set_xlabel('Cloud Cover')
ax1.set_ylabel('频数')

# 2. Season直方图
ax2 = plt.subplot(gs[0, 1])
sns.histplot(data1['Season'], color=colors[1], ax=ax2)
ax2.set_title('Season分布直方图')
ax2.set_xlabel('Season')
ax2.set_ylabel('频数')

# 3. Location直方图
ax3 = plt.subplot(gs[1, 0])
sns.histplot(data1['Location'], color=colors[2], ax=ax3)
ax3.set_title('Location分布直方图')
ax3.set_xlabel('Location')
ax3.set_ylabel('频数')

# 4. Weather Type直方图
ax4 = plt.subplot(gs[1, 1])
sns.histplot(data1['Weather Type'], color=colors[3], ax=ax4)
ax4.set_title('Weather Type分布直方图')
ax4.set_xlabel('Weather Type')
ax4.set_ylabel('频数')

# 创建编码器
le=LabelEncoder()
encoders = {}  # 用于保存目标列的编码器

data_one_hot = data1.copy()
target_columns = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
for col in target_columns:
    if col=='Weather Type':
        le = LabelEncoder()
        data_one_hot[col] = le.fit_transform(data_one_hot[col])
        encoders[col] = le  # 保存编码器
    else:
        data_one_hot[col] = le.fit_transform(data_one_hot[col])

print(data_one_hot[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']].corr().round(2))

# 使用孤立森林检测异常值
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(data_one_hot[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']])

# 预测异常值 (-1表示异常，1表示正常)
data_one_hot['anomaly'] = clf.predict(data_one_hot[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']])

# 可视化异常值
plt.figure(figsize=(10, 6))
plt.scatter(data_one_hot[data_one_hot['anomaly'] == 1].index, data_one_hot[data_one_hot['anomaly'] == 1]['Temperature'], c='blue', label='正常')
plt.scatter(data_one_hot[data_one_hot['anomaly'] == -1].index, data_one_hot[data_one_hot['anomaly'] == -1]['Temperature'], c='red', label='异常')
plt.title('使用孤立森林检测特征Temperature的异常值')
plt.legend()
plt.show()

# 删除异常值
data_cleaned_iforest = data_one_hot[data_one_hot['anomaly'] == 1].drop('anomaly', axis=1)


# 初始化标准化器
scaler=StandardScaler()
# 初始化归一化器（缩放到[0,1]）
normalizer = MinMaxScaler(feature_range=(0, 1))

# 对DataFrame中的数值列进行标准化
# 注意:需线选择数值列，避免处理类别变量
data_standardized=data_cleaned_iforest.copy()
data_standardized[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']]=scaler.fit_transform(data_standardized[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']])

# 对数值列进行归一化
data_normalized = data_standardized.copy()
data_normalized[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']] = normalizer.fit_transform(data_normalized[['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Visibility (km)']])

target_names=[]
for i, class_name in enumerate(encoders['Weather Type'].classes_):
    print(f"编码 {i} → {class_name}")
    target_names.append(class_name)
print(target_names)


x = data_normalized.iloc[:, :-1]  # 取除最后一列之外的所有列
y = data_normalized.iloc[:, -1:].values.ravel()  # 取最后一列并展平为一维数组

# 数据探索
print(f"数据集形状: {x.shape}")

# 创建决策树模型实例
model = DecisionTreeClassifier(random_state=42)

# 定义网格搜索参数空间
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15, 20]
}

# 设置10折交叉验证的网格搜索
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1  # 使用所有CPU核心
)

# 执行网格搜索
grid_search.fit(x, y)

# 输出最优参数和最佳模型性能
print("\n网格搜索结果:")
print(f"最优参数: {grid_search.best_params_}")
print(f"交叉验证最佳准确率: {grid_search.best_score_:.3f}")

# 获取最优参数的模型
best_model = grid_search.best_estimator_

# 输出最优模型的详细参数
print("\n最优模型参数详情:")
for param, value in best_model.get_params().items():
    if param in param_grid:
        print(f"{param}: {value}")

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# 训练最优模型
best_model.fit(x_train, y_train)

# 在测试集上进行预测
y_pred = best_model.predict(x_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.2f}")

# 查看详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

