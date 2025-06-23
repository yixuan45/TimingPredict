import pandas as pd


class TimeSeries(object):
    """
    时间序列数据预处理类，用于加载、分割和预处理时间序列数据
    """

    def __init__(self, filename, train_size=0.7):
        """
        初始化TimeSeries类，加载并分割时间序列数据
        :param filename: csv文件路径
        :param train_size: 训练集比例，范围[0.0,1.0]
        """
        self.data = pd.read_csv(filename)
        # 假设数据是按天记录的，从 2023-01-01 开始
        start_date = '2010-01-01'

        # 获取 Store_Sales 列的长度
        n_rows = len(self.data['Store_Sales'])

        # 生成日期序列
        self.data['time'] = pd.date_range(start=start_date, periods=n_rows, freq='D')
        self.data['time'] = pd.to_datetime(self.data['time'])

        # 确保列名正确
        if len(self.data.columns) >= 2:
            self.data = self.data[['time', 'Store_Sales']]  # 只需要这两列数据
        else:
            raise ValueError("数据至少需要包含'time'和'Store_Sales'这两列数据")

        # 因为这里没有没有日期，只是用的ID作为一个索引值
        self.data.set_index('time', inplace=True)

        # 验证train_size的大小
        if not (0.0 < train_size < 1.0):  # 这里需要保证train_size的合理性
            raise ValueError("train_size必须要在0.0到1.0之间")

        # 接下来是分割数据
        n_rows = len(self.data)  # 获取数据的行数
        split_idx = int(n_rows * train_size)
        self.train = self.data.iloc[:split_idx].copy()
        self.test = self.data.iloc[split_idx:].copy()

    def set_scale(self, factor=1.0):
        """
        缩放时间序列数据。
        :param factor: 缩放因子
        :return: 新的TimeSeries对象，包含缩放后的数据
        """
        if factor <= 0.0:
            raise ValueError("缩放因子不能为0")

        # 创建新对象
        scaled_ts = TimeSeries.__new__(TimeSeries)  # 这行代码的作用是创建一个新的 TimeSeries 类实例，
        # 但不调用该类的 __init__ 方法。这是一种在 Python 中创建对象的底层方式
        scaled_ts.data = self.data.copy()
        scaled_ts.data['Store_Sales'] /= factor

        scaled_ts.train = self.train.copy()
        scaled_ts.train['Store_Sales'] /= factor

        scaled_ts.test = self.test.copy()
        scaled_ts.test['values'] /= factor

        return scaled_ts
