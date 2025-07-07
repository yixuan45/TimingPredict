import torch
from sklearn.model_selection import learning_curve

from 序列模型.序列模型 import batch_size

torch.manual_seed(1024)

x = torch.randn(100, 200, 300)
x = (x - torch.mean(x)) / torch.std(x)
epsilon = torch.randn(x.shape)
y = 10 * x + 5 + epsilon

import matplotlib.pyplot as plt

plt.scatter(x, y)

from sklearn import linear_model

m = linear_model.LinearRegression()
m.fit(x.view(-1, 1), y)
m.coef_, m.intercept_

#### 梯度下降法
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self):
        # 定义模型参数
        super(Linear, self).__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        # 前向传播
        return self.a * x + self.b

    def string(self):
        return f'y={self.a.item():.2f}*x + {self.b.item():.2f}'


m = Linear()
m(x)

list(m.parameters())

import torch.optim as optim

loss_list = []
learning_rate = 0.1
model = Linear()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for t in range(20):
    y_pred = model(x)
    # 定义损失
    loss = (y - y_pred).pow(2).mean()
    loss_list.append(loss.item())
    # 在计算当前梯度的时候一定要将上一步梯度清空
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()
    print(model.string())

plt.plot(loss_list)

import torch.optim as optim

loss_list = []
learning_rate = 0.1
model = Linear()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for t in range(20):
    y_pred = model(x)
    # 定义损失
    loss = (y - y_pred).pow(2).mean()
    loss_list.append(loss.item())
    # 在计算当前梯度的时候一定要将上一步梯度清空
    # optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新模型参数
    # optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad = None
    print(model.string())

plt.plot(loss_list)

### 随机梯度下降法
loss_list = []
batch_size = 20
learning_rate = 0.1
model = Linear()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for t in range(20):
    ix = (t * batch_size) % len(x)
    xx = x[ix:ix + batch_size]
    yy = y[ix:ix + batch_size]
    yy_pred = model(xx)
    # 定义损失
    loss = (yy - yy_pred).pow(2).mean()
    loss_list.append(loss.item())
    # 在计算当前梯度的时候一定要将上一步梯度清空
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()
    print(model.string())

plt.plot(loss_list)

#### 张量的基本操作
torch.zeros(2, 3)

torch.randn(4, 2)

