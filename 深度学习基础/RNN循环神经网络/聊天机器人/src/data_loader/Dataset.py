import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import random
import os
from tqdm import tqdm

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChatDataset(Dataset):
    def __init__(self, questions, answers, word_to_idx, idx_to_word, max_seq_len=20):
        self.questions = questions
        self.answers = answers
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_seq_len = max_seq_len
        self.vocab_size = len(word_to_idx)

        # 添加特殊标记
        self.SOS_token = '<SOS>'  # 句子开始标记
        self.EOS_token = '<EOS>'  # 句子结束标记
        self.PAD_token = '<PAD>'  # 填充标记
        self.UNK_token = '<UNK>'  # 未知词标记

        # 将特殊标记添加到词汇表
        for token in [self.SOS_token, self.EOS_token, self.PAD_token, self.UNK_token]:
            if token not in word_to_idx:
                self.word_to_idx[token] = len(self.word_to_idx)
                self.idx_to_word[len(self.word_to_idx)] = token

        # 预处理数据
        self.processed_data = self._process_data()

    def _process_data(self):
        processed = []
        for q, a in zip(self.questions, self.answers):
            # 简单的文本清理
            q_clean = self._clean_text(q)
            a_clean = self._clean_text(a)

            # 转换为索引序列
            q_indices = self._text_to_indices(q_clean)
            a_indices = self._text_to_indices(a_clean)

            processed.append((q_indices, a_indices))

        return processed

    @staticmethod
    def _clean_text(text):
        # 简单的文本清理:转化为小写，转移标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # \w 是元字符，等价于 [a-zA-Z0-9_]，表示所有字母、数字和下划线（单词字符），
        # \s 是元字符，表示所有空白字符（空格 、制表符 \t、换行符 \n 等）。
        return text

    def _text_to_indices(self, text):
        # 将文本转化为索引序列
        words = text.split()  # 作为空白字符进行切割
        indices = [self.word_to_idx.get(word, self.word_to_idx[self.UNK_token]) for word in words]
        return indices

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        q, a = self.processed_data[idx]
        return torch.tensor(q), torch.tensor(a)
