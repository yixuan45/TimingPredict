"""
analysis of Sales data
"""
from time_series import TimeSeries

# Import for data visualization
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from matplotlib import dates as mpld

# Seasonal Decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Holt-Winters or Triple Exponential Smoothing model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

register_matplotlib_converters()

ts = TimeSeries('./Stores.csv', train_size=0.8)

print("Sales Data\n")
print(ts.data.describe())

print("\nHead and Tail of the time series\n")
print(ts.data.head(5).iloc[:, 0:])
print(ts.data.tail(5).iloc[:, 0:])

# Plot of raw time series data
plt.plot(ts.data.index, ts.data.values)
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.title("Sales Data Analysis (2010-2012)")
plt.xlabel("Time")
plt.ylabel("Store_Sales")
plt.show()

"""
时间序列的季节性分解
季节性分解是一种用于将时间序列的构成要素分解为以下几部分的方法：
水平值 - 序列中的平均值。
趋势 - 序列中呈现上升或下降趋势的值。
季节性 - 序列中重复出现的短期周期。
噪声 - 序列中的随机变动。
对这些构成要素分别进行分析，能为模型选择提供更有价值的见解。
"""

result_add = seasonal_decompose(
    ts.data.iloc[:, 0], period=12, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(
    ts.data.iloc[:, 0], period=12, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

"""
根据描述，选择 Holt-Winters 加法模型 是合理的，原因如下：

恒定的季节性成分：季节性波动幅度相对稳定，适合用加法模型（model='additive'）。
上升趋势：Holt-Winters 模型可以很好地捕捉趋势和季节性成分。
Holt-Winters 模型的适用场景：
数据存在趋势和季节性。
季节性成分的波动幅度相对稳定（加法模型）或随趋势变化（乘法模型）。
"""
# Training the model
model = ExponentialSmoothing(ts.train, trend='additive',
                             seasonal='additive', seasonal_periods=12).fit(damping_slope=1)
plt.plot(ts.train.index, ts.train, label="Train")
plt.plot(ts.test.index, ts.test, label="Actual")

# Create a 5 year forecast
plt.plot(model.forecast(60), label="Forecast")

plt.legend(['Train', 'Actual', 'Forecast'])
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.title("Sales Data Analysis (2010-2012)")
plt.xlabel("Time")
plt.ylabel("Sales (x1000)")
plt.show()

"""
模型的验证
让我们对加法模型和乘法模型做一个简单的对比分析。
"""
ts = TimeSeries('./Stores.csv', train_size=0.8)

# Additive model
model_add = ExponentialSmoothing(
    ts.data.iloc[:, 0], trend='additive', seasonal='additive', seasonal_periods=12, damped=True).fit(damping_slope=0.98)
prediction = model_add.predict(
    start=ts.data.iloc[:, 0].index[0], end=ts.data.iloc[:, 0].index[-1])
plt.plot(ts.data.iloc[:, 0].index, ts.data.iloc[:, 0], label="Train")
plt.plot(ts.data.iloc[:, 0].index, prediction, label="Model")
plt.plot(model_add.forecast(60))

plt.legend(['Actual', 'Model', 'Forecast'])
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.title("Sales Data Analysis (2010-2012)")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()

# Multiplicative model
model_mul = ExponentialSmoothing(
    ts.data.iloc[:, 0], trend='additive', seasonal='multiplicative', seasonal_periods=12, damped=True).fit()
prediction = model_mul.predict(
    start=ts.data.iloc[:, 0].index[0], end=ts.data.iloc[:, 0].index[-1])
plt.plot(ts.data.iloc[:, 0].index, ts.data.iloc[:, 0], label="Train")
plt.plot(ts.data.iloc[:, 0].index, prediction, label="Model")
plt.plot(model_mul.forecast(60))
plt.legend(['Actual', 'Model', 'Forecast'])
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.title("Sales Data Analysis (2010-2012)")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()

print(model_add.summary())
print(model_mul.summary())
if __name__ == '__main__':
    pass
