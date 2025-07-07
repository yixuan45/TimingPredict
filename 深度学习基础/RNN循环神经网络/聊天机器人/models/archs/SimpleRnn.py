from torch import nn


class SimpleRnn(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(SimpleRnn, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 纯RNN层(不使用LSTM/GRU)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # 输出层，将隐藏状态映射到词汇表大小的空间
        self.fc=nn.Linear(hidden_size, hidden_size)

    def forward(self, input_seq,hidden=None):
        # 嵌入输入序列
        embedding = self.embedding(input_seq)

