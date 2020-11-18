"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'
import torch
import math
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
from setting import LAYERS,D_MODEL,D_FF,DROPOUT,H_NUM,TGT_VOCAB,SRC_VOCAB

from setting import DEVICE

"""
Encoder 大体架构
[
    Embedding + PositionalEncoding 
        X (batch_size, seq_len, embed_dim)
    self attention, attention mask 
        X_attention = SelfAttention(Q,K,V)
    Layer Normalization, Residual
        X_attention = LayerNorm(X_attention)
        X_attention = X + X_attention
    Feed Forward
        X_hidden = Linear(Activate(Linear(X_attention)))
    Layer Normalization, Residual
        X_hidden = LayerNorm(X_hidden)
        X_hidden = X_attention + X_hidden  
]
"""

"""模型各层定义"""
class Embeddings(nn.Module):
    """
    模型中，再两个embedding层和pre-softmax线性变换层之间共享相同的权重矩阵

    embedding层中，将权重乘以 sqrt(d_model)
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    """
    PositionalEncoding位置编码是固定的
    生成之后，整个训练过程不改变
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: embedding的维数
        :param dropout: dropout概率
        :param max_len: 最大句子长度
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个 max_len, d_model 维度的全零矩阵, 用来存放位置向量
        # 每一行都是一个位置下标
        pe = torch.zeros(max_len, d_model, device=DEVICE)

        # (max_len,1)
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)

        # 使用exp log 实现sin/cos公式中的分母
        div_term = torch.exp(torch.arange(0,d_model,2,device=DEVICE) *
                             (-math.log(10000.0) / d_model))
        # TODO

        # 填充 max_len, d_model 矩阵
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 方便和 一个batch的句子所有词embedding批量相加
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return x

def attention(query, key, value, mask=None, dropout=None):
    """
    h代表每个多少个头部，这里使矩阵之间的乘机操作，对所有的head进行同时计算
    :param query: batch_size, h, sequence_len, embedding_dim
    :param key: batch_size, h, sequence_len, embedding_dim
    :param value: batch_size, h, sequence_len, embedding_dim
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    # 将key的最后两个维度互换才能和query矩阵相乘
    # 将相乘结果除以一个归一化的值
    # scores矩阵维度： batch_size,h,sequence_len, sequence_len
    # scores矩阵中的每一行代表着长度为sequence_len的句子中每个单词与其他单词的相似度
    scores = torch.matmul(query,key.transpose(-2,-1) / math.sqrt(d_k))
    # 归一化 ，归一化之前要进行mask操作，防止填充的0字段影响
    if mask is not None:
        # 将填充的0值变为很小的负数，根据softmax exp^x的图像，当x的值很小的时候，值很接近0
        # 补充，当值为0时， e^0=1
        scores = scores.masked_fill(mask==0,-1e9)  # mask掉矩阵中为0的值
    p_attn = F.softmax(scores,dim=-1) # 每一行 softmax #TODO 计算每一行的softmax，按列

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回注意力和 value的乘机，以及注意力矩阵
    # batch_size,h,sequence_len, sequence_len * batch_size, h, sequence_len, embed_dim
    # 返回结果矩阵维度 batch_size,h,sequence_len,embed_dim
    return torch.matmul(p_attn, value), p_attn


class MultiHeaderAttention(nn.Module):
    """
    首先初始化三个权重矩阵 W_Q, W_K, W_V
    将embed x 与 权重矩阵相乘，生成Q,K,V
    将Q K V分解为多个head attention
    对不同的QKV，计算Attention（softmax(QK^T/sqrt(d_model))V）

    输出 batch_size, sequence_len, embed_dim
    """
    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: head的数目
        :param d_model: embed的维度
        :param dropout:
        """
        super(MultiHeaderAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h   # 每一个head的维数
        self.h = h  # head的数量

        # 定义四个全连接函数 WQ,WK,WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存attention结果
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # query维度(transpose之后)：batch_size, h, sequence_len, embedding_dim/h
        query, key, value = [l(x).view(nbatches, -1,self.h, self.d_k).transpose(1,2)
                             for l,x in zip(self.linears, (query, key, value))]
        # 对query key value 计算 attention
        # attention 返回最后的x 和 atten weight
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多个头的注意力矩阵concat起来
        # 输入：x shape: batch_size, h, sequence_len, embed_dim/h(d_k)
        # 输出：x shape: batch_size, sequence_len, embed_dim
        x = x.transpose(1,2).contiguous().view(nbatches, -1,self.h*self.d_k)


        return self.linears[-1](x)  # batch_size, sequence_len, embed_dim

class LayerNorm(nn.Module):
    """
    Normalization https://zhuanlan.zhihu.com/p/33173246
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 按照最后一个维度计算均值和方差 ，embed dim
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x-mean) / (torch.sqrt(std**2 + self.eps)) + self.b_2


class SublayerConnection(nn.Module):
    """
    sublayerConnection把Multi-Head Attention和Feed Forward层连在一起

    组合LayerNorm和Residual
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm 和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """对每个position采取相同的操作
    初始化参数：
        d_model: embedding维数
        d_ff: 隐层网络单元数
        dropout
    前向传播参数: x 维数 (batch_size, sequence_len, embed_dim)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) #TODO


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self,x,mask):
        # 将embedding输入进行multi head attention
        # 得到 attention之后的结果
        """
        lambda x:self.... 是一个函数对象，参数是x
        """
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)

def clones(module, N):
    """
    将传入的module深度拷贝N份
    参数不共享
    :param module: 传入的模型 ex:nn.Linear(d_model, d_model)
    :param N: 拷贝的N份
    :return: nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder会使用Encoder的K，V
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # clone了三个，Decoder有三块
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory
        # self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个decoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


class LabelSmoothing(nn.Module):
    """标签平滑处理"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        """
        损失函数是KLDivLoss，那么输出的y值得是log_softmax
        具体请看pytorch官方文档，KLDivLoss的公式
        """
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

"""损失函数"""
class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


"""优化器"""
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))




""" 对模型进行初始化"""

def make_model(src_vocab, tgt_vocab, N=LAYERS, d_model=D_MODEL, d_ff=D_FF, h=H_NUM, dropout=DROPOUT):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeaderAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)





