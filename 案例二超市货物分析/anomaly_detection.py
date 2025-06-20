"""
Anomaly Detection Using Brutlag algorithm
该算法特别适用于具有季节性特征的时间序列数据，如：

温度、降水等气象数据异常检测
能源消耗、电力负荷异常识别
销售数据、网站流量的异常波动监测
工业设备运行参数的异常检测

通过这种方法，可以有效识别出偏离正常季节性模式的异常点，对于气象分析、趋势研究和异常事件监测具有重要价值。
"""
from time_series import TimeSeries

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# imports for data visualization
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib import dates as mpld

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

ts = TimeSeries('./Stores.csv', train_size=0.7)
plt.plot(ts.data.iloc[-100:, 0].index, ts.data.iloc[-100:, 0])
plt.gcf().autofmt_xdate()
plt.title('Store_Sales')
plt.xlabel('Time')
plt.ylabel('Store_Sales')
plt.show()

model = ExponentialSmoothing(
    ts.train, trend='additive', seasonal='additive').fit()
prediction = model.predict(
    start=ts.data.iloc[:, 0].index[0], end=ts.data.iloc[:, 0].index[-1])

"""Brutlag Algorithm"""
PERIOD = 12  # The given time series has seasonal_period=12
GAMMA = 0.3684211  # the seasonility component
SF = 1.96  # brutlag scaling factor for the confidence bands. # 置信区间
UB = []  # upper bound or upper confidence band
LB = []  # lower bound or lower confidence band
difference_array = []
dt = []
difference_table = {
    "actual": ts.data.iloc[:, 0], "predicted": prediction, "difference": difference_array, "UB": UB, "LB": LB}

"""Calculatation of confidence bands using brutlag algorithm"""
for i in range(len(prediction)):
    diff = ts.data.iloc[:, 0][i] - prediction[i]
    if i < PERIOD:
        dt.append(GAMMA * abs(diff))
    else:
        dt.append(GAMMA * abs(diff) + (1 - GAMMA) * dt[i - PERIOD])

    difference_array.append(diff)
    UB.append(prediction[i] + SF * dt[i])
    LB.append(prediction[i] - SF * dt[i])

print("\nDifference between actual and predicted\n")
difference = pd.DataFrame(difference_table)
print(difference)

"""Classification of data points as either normal or anomaly"""
normal = []
normal_date = []
anomaly = []
anomaly_date = []

for i in range(len(ts.data.iloc[:, 0].index)):
    if (UB[i] <= ts.data.iloc[:, 0][i] or LB[i] >= ts.data.iloc[:, 0][i]) and i > PERIOD:
        anomaly_date.append(ts.data.index[i])
        anomaly.append(ts.data.iloc[:, 0][i])
    else:
        normal_date.append(ts.data.index[i])
        normal.append(ts.data.iloc[:, 0][i])

anomaly = pd.DataFrame({"date": anomaly_date, "value": anomaly})
anomaly.set_index('date', inplace=True)
normal = pd.DataFrame({"date": normal_date, "value": normal})
normal.set_index('date', inplace=True)

print("\nThe data points classified as anomaly\n")
print(anomaly)

"""
Plotting the data points after classification as anomaly/normal.
Data points classified as anomaly are represented in red and normal in green.
"""
plt.plot(normal.index, normal, 'o', color='green')
plt.plot(anomaly.index, anomaly, 'o', color='red')

# Ploting brutlag confidence bands
plt.plot(ts.data.iloc[:, 0].index, UB, linestyle='--', color='grey')
plt.plot(ts.data.iloc[:, 0].index, LB, linestyle='--', color='grey')

# Formatting the graph
plt.legend(['Normal', 'Anomaly', 'Upper Bound', 'Lower Bound'])
plt.gcf().autofmt_xdate()
plt.title("Average Temperature of India (2000-2018)")
plt.xlabel("Time")
plt.ylabel("Temparature (°C)")
plt.show()

if __name__ == '__main__':
    pass
