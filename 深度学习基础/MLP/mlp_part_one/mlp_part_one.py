import torch

torch.manual_seed(1024)


# 定义线性模型何sigmoid函数

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

    def __call__(self, x):
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


l = Linear(3, 4)
x = torch.randn(5, 3)
l(x).shape

l.parameters()


class Sigmoid:
    def __call__(self, x):
        self.out = torch.sigmoid(x)
        return self.out

    @staticmethod
    def parameters():
        return []


s = Sigmoid()
X = torch.randn(5, 3)
s(x).shape


class Perceptron:
    def __init__(self, in_features):
        self.ln = Linear(in_features, 1)
        self.f = Sigmoid()

    def __call__(self, x):
        # x:(B,in_features)
        self.out = self.f(self.ln(x))  # (B,1)
        return self.out

    def parameters(self):
        return self.ln.parameters() + self.f.parameters()


class LogitRegression:
    # input:(B,in_features)
    # output：（B，2）
    def __init__(self, in_features):
        self.pos = Linear(in_features, 1)
        self.neg = Linear(in_features, 1)

    def __call__(self, x):
        # x:(B,in_features)
        self.out = torch.concat((self.pos(x), self.neg(x)), dim=-1)  # (B,2)
        return self.out

    def parameters(self):
        return self.pos.parameters() + self.neg.parameters()


lr = LogitRegression(3)
x = torch.randn(5, 3)
lr(x).shape
lr.parameters()

logits = lr(x)
logits

import torch.nn.functional as F

probs = F.softmax(logits, dim=-1)

pred = torch.argmax(probs, dim=-1)
print(probs)
print(pred)

logits
pred
loss = F.cross_entropy(logits, pred)
loss

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data = make_blobs(200, centers=[[-2, -2], [2, 2]])
x, y = data
plt.scatter(x[:, 0], y[:, 1])

batch_size = 20
max_steps = 2000
learning_rate = 0.01
x, y = torch.tensor(data[0]).float(), torch.tensor(data[1])
lr = LogitRegression(2)
lossi = []

for t in range(max_steps):
    ix = (t * batch_size) % len(x)
    xx = x[ix:ix + batch_size]
    yy = y[ix:ix + batch_size]  # (20)
    logits = lr(xx)  # (20,2)
    loss = F.cross_entropy(logits, yy)
    loss.backward()
    with torch.no_grad():
        for p in lr.parameters():
            p -= learning_rate * p.grad
            p.grad = None

    if t % 200 == 0:
        print(f'step{t}, loss{loss.item()}')
    lossi.append(loss.item())

plt.plot(lossi)
