import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
%matplotlib inline


torch.manual_seed(12046)

# 一些超参数
context_length=10
learning_rate=0.01
eval_interval=10
batch_size=1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

raw_datasets = load_dataset('code_search_net', 'python')
datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
# 通过索引提取datasets数据的时候，返回一个dict，其中的value是一个字符串
print(datasets[8]['whole_func_string'])
# 当传入的是一个数组时，返回的依然是一个dict，但其中的value是一个列表
print(datasets[8: 10]['whole_func_string'])

class char_tokenizer(object):
    def __init__(self,data,begin_ind=0,end_ind=1):
        # 数据中出现的所有字符串构成字典
        chars=sorted(list(set(''.join(data))))
        # 预留两个位置给开头和结尾的特殊字符
        self.char2ind={s:i+2 for i,s in enumerate(chars)}
        print(self.char2ind)
        self.char2ind['<|b|>'] = begin_ind
        self.char2ind['<|e|>'] = end_ind
        self.begin_ind=begin_ind
        self.end_ind=end_ind
        # 将数字映射到字符
        self.ind2char={i:s for s ,i in self.char2ind.items()}

    def encode(self,text):
        """
        编码参数
        ------
        text：str,文本
        """
        return [self.char2ind[c] for c in text]

    def decode(self,enc):
        """
        解码参数
        ------
        enc :int or list[int]
        """
        if isinstance(enc,int):
            return self.ind2char[enc]
        return [self.ind2char[i] for i in enc]

# 举例验证分词器
tok = char_tokenizer(datasets['whole_func_string'])
example_text = 'def postappend(self):'
''.join(tok.decode(tok.encode(example_text))), len(tok.char2ind)

def autoregressive_trans(text,tokenizer,context_length=context_length):
    '''
    将文本转换成一系列的训练数据
    参数
    ----
    text ：str，文本
    tokenizer ：分词器
    context_length ：int，背景文本的长度
    返回
    ----
    inputs ：list[list[int]]，背景文本（特征）
    labels ：list[list[int]]，预测标签
    '''
    inputs,labels=[],[]
    b_ind=tokenizer.begin_ind
    e_ind=tokenizer.end_ind
    enc=tokenizer.encode(text)
    # 增加开始和结尾的特殊字符
    x=[b_ind] *context_length +enc+[e_ind]
    print(x)
    for i in range(len(x)-context_length):
        inputs.append(x[i:i+context_length])
        labels.append(x[i+context_length])
    return inputs,labels

# 举例展示自回归模式的训练数据
inputs, labels = autoregressive_trans(example_text, tok)
for a, b in zip(inputs, labels):
    print(''.join(tok.decode(a)), '--->',  tok.decode(b))


# 将数据分为训练集和测试集
tokenized=datasets.train_test_split(test_size=0.1,seed=1024,shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=device)

tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape