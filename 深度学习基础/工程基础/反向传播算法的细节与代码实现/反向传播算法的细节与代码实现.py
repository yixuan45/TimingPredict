class ScalarTmp:

    def __init__(self, value, prevs=[], op=None, label=''):
        # 定义节点的值
        self.value = value
        # 定义直接的前序节点
        self.prevs = prevs
        # 定义运算符号：op或者变量名label
        self.op = op
        self.label = label

    def __repr__(self):
        return f'{self.value} | {self.op} | {self.label}'

    def __add__(self, other):
        # self_other触发这个函数
        value = self.value + other.value
        prevs = [self, other]
        output = ScalarTmp(value, prevs, op='+')
        return output

    def __mul__(self, other):
        # self*other出发这个函数
        value = self.value * other.value
        prevs = [self, other]
        output = ScalarTmp(value, prevs, op='*')
        return output
