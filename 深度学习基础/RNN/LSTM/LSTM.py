import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
%matplotlib
inline

torch.manual_seed(12046)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        长短期记忆网络的神经元
        参数
        ---
        input_size:int,输入数据的特征长度
        hidden_size:int,隐藏状态的特征长度
        '''
        super(LSTMCell, self).__init__()
        self.input_size = input_size  # input_size :I
        self.hidden_size = hidden_size  # hidden_size :H
        combined_size = self.input_size + self.hidden_size  # combined_size :I+H
        # 定义输入们的线性部分
        self.in_gate = nn.Linear(combined_size, self.hidden_size)
        # 定义遗忘门的线性部分
        self.forget_gate = nn.Linear(combined_size, self.hidden_size)
        # 定义备选细胞状态的线性部分
        self.new_cell_gate = nn.Linear(combined_size, self.hidden_size)
        # 定义输入门的线性部分
        self.out_gate = nn.Linear(combined_size, self.hidden_size)

    def forward(self, inputs, state=None):
        '''
        向前传播
        参数
        ---
        inputs:torch.FloatTensor,输入数据，形状为(B,I),其中B表示批量大小，I表示文字特征的长度(input_size)
        state ：tuple(torch.FloatTensor, torch.FloatTensor)
            (隐藏状态，细胞状态)，两个状态的形状都为(B, H)，其中H表示隐藏状态的长度（hidden_size）
        返回
        ----
        hs ：torch.FloatTensor，隐藏状态，形状为(B, H)
        cs ：torch.FloatTensor，细胞状态，形状为(B, H)
        '''
        B, _ = inputs.shape
        if state is None:
            state = self.init_state(B, inputs.device)
        hs, cs = state
        combined = torch.cat((inputs, hs), dim=1)  # (B,I+H)
        # 输入门
        ingate = F.sigmoid(self.in_gate(combined))  # (B,    H)
        # 遗忘门
        forgetgate = F.sigmoid(self.forget_gate(combined))  # (B,     H)
        # 输入门
        outgate = F.sigmoid(self.out_gate(combined))  # (B,     H)
        # 更新细胞状态
        ncs = F.tanh(self.new_cell_gate(combined))
        cs = (forgetgate * cs) + (ingate * ncs)
        # 更新隐藏状态
        hs = outgate * F.tanh(cs)
        return hs, cs

    def init_state(self, B, device):
        # 默认的隐藏状态和细胞状态全部都等于0
        cs = torch.zeros((B, self.hidden_size), device=device)
        hs = torch.zeros((B, self.hidden_size), device=device)
        return hs, cs


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        单层的长短期记忆网络（支持批量计算）
        参数
        ---
        input_size:int,输入数据的特征长度
        hidden_size:int,隐藏状态的特征长度
        '''
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCell(self.input_size, self.hidden_size)

    def forward(self, inputs, state=None):
        '''
        向前传播
        参数
        ----
        inputs ：torch.FloatTensor
            输入数据的集合，形状为(B, T, C)，其中B表示批量大小，T表示文本长度，C表示文字特征的长度（input_size）
        state ：tuple(torch.FloatTensor, torch.FloatTensor)
            (初始的隐藏状态，初始的细胞状态)，两个状态的形状都为(B, H)，其中H表示隐藏状态的长度（hidden_size）
        返回
        ----
        hidden ：torch.FloatTensor，所有隐藏状态的集合，形状为(B, T, H)
        '''
        re = []
        B, T, C = inputs.shape
        inputs = inputs.transpose(0, 1)  # (T,B,C)
        for i in range(T):
            state = self.lstm(inputs[i], state)
            # 只记录隐藏状态，state[0]的形状为(B,H)
            re.append(state[0])
        result_tensor=torch.stack(re,dim=0) # (T,B,H)
        return result_tensor.transpose(0,1) # (B,T,H)

