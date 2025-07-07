import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

torch.manual_seed(1024)


class Linear:
    # input:(B,in_features)
    # output:(B,out_features)
    def __init__(self, in_features, out_features, bias=True):
        # 对于模型参数的初始化，故意没有做优化
        self.weight = torch.randn(in_features, out_features, requires_grad=True)  # (in_features,out_features)
        if bias:
            self.bias = torch.randn(out_features, requires_grad=True)  # (out_features)
        else:
            self.bias = None

    def __call__(self, x):  # 方法让类的实例可以像函数一样被调用
        # x:  (B,in_features)
        # self.weight:(in_features,out_features)
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        # 返回模型参数
        if self.bias is not None:
            return [self.weight, self.bias]
        return self.weight


class Sigmoid:
    def __call__(self, x):
        self.out = torch.sigmoid(x)
        return self.out

    @staticmethod
    def parameters():
        return []


class Sequential:
    def __init__(self, layers):
        # layers表示的模型组件，比如线性模型，比如sigmoid
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        self.out = x
        return self.out

    def parameters(self):
        # k=[]
        # for layer in self.layers():
        #     for p in layer.parameters():
        #          k.append(p)
        return [p for layer in self.layers for p in layer.parameters()]

    def predict_proba(self, x):
        # 计算概率预测
        if isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        logits = self(x)  # 等价于 self.__call__(x)
        self.prob = F.softmax(logits, dim=-1).detach().numpy()
        return self.prob


# x:(B,2)
# mlp:[4,4,2]
model = Sequential([
    Linear(2, 4), Sigmoid(),  # （B，4）
    Linear(4, 4), Sigmoid(),  # （B，4）
    Linear(4, 2)  # （B，2）
])
x = torch.randn(3, 2)
model(x)

model.predict_proba(x)


def draw_data(data):
    """
    数据可视化
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    x, y = data
    label1 = x[y > 0]
    ax.scatter(label1[:, 0], label1[:, 1], marker='o')
    label0 = x[y == 0]
    ax.scatter(label0[:, 0], label0[:, 1], marker='^', color='k')
    return ax


def draw_model(ax, model):
    """
    将模型的分离超平面可视化
    """
    x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    x2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = model.predict_proba(np.c_[x1.ravel(), x2.ravel()])[:, 1]
    y = y.reshape(x1.shape)
    ax.contourf(x1, x2, y, levels=[0, 0.5], colors=['gray'], alpha=0.4)
    return ax


data = make_moons(200, noise=0.05)
draw_data(data)

batch_size = 20
max_steps = 2000
learning_rate = 0.01
x, y = torch.tensor(data[0]).float(), torch.tensor(data[1])
lossi = []

for epoch in range(max_steps):
    ix = (epoch * batch_size) % len(x)
    xx = x[ix:ix + batch_size]
    yy = y[ix:ix + batch_size]
    logits = model.predict_proba(xx)
    loss = F.cross_entropy(logits, yy)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= -learning_rate * p.grad
            p.grad.zero_()
    if epoch % 200 == 0:
        print(f'step{epoch}, loss{loss.item()}')
    lossi.append(loss.item())
