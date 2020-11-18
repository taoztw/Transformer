"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'

from nltk import word_tokenize
from utils import get_word_dict, subsequent_mask
import torch
import numpy as np
from torch.autograd import Variable
from setting import SAVE_FILE,DEVICE,LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, TGT_VOCAB, SRC_VOCAB


"""
单个句子输入，单个句子翻译输出
"""

# from train import model  # 可以通知train直接导入初始化ok的模型，也可以使用下面的初始化模型的方式
# 初始化模型
def init_model():
    from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, TGT_VOCAB, SRC_VOCAB
    from model import make_model
    # 模型的初始化
    model = make_model(
        SRC_VOCAB,
        TGT_VOCAB,
        LAYERS,
        D_MODEL,
        D_FF,
        H_NUM,
        DROPOUT
    )
    return model
model = init_model()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cn_idx2word, cn_word2idx, en_idx2word, en_word2idx = get_word_dict()

model.load_state_dict(torch.load(SAVE_FILE,map_location=torch.device('cpu')))


def sentence2id(sentence):
    """
    word2id，将句子单词转为id表示
    测试范例：
    # >>> print(sentence2id('I am a boy'))
    [['2', '5', '90', '9', '192', '3']]
    :param sentence:一个英文句子
    :return: 双层列表，将句子中每个单词使用id表示
    """
    en = []
    en.append(['BOS'] + word_tokenize(sentence.lower()) + ['EOS'])

    sentence_id = [[int(en_word2idx.get(w,0)) for w in e] for e in en]
    return sentence_id

def src_handle(X):


    """
    将句子id列表转换为tensor，并且生成输入的mask矩阵
    :param X: 单词列表id的list
    :return: 单词列表id的list和对输入的mask
    """
    src = torch.from_numpy(np.array(X)).long().to(DEVICE)
    src_mask = (src != 0).unsqueeze(-2)
    return src, src_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def output(out):
    translation = []
    # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
    for j in range(1, out.size(1)): # 生成的最大程度的序列
        # 获取当前下标的输出字符

        sym = cn_idx2word[out[0, j].item()]
        # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
        if sym != 'EOS':
            translation.append(sym)
        else:
            break
    # 打印模型翻译输出的中文句子结果
    print("translation: %s" % " ".join(translation))
    return ''.join(translation)

def machine_translate(sentence):
    """
    实现机器翻译
    :param sentence: 输入一个句子
    :return: 输出机器翻译的结果
    """
    src,src_mask = src_handle(sentence2id(sentence))
    out = greedy_decode(model,src,src_mask,max_len=50,start_symbol=int(cn_word2idx.get('BOS')))
    cn_result = output(out)
    return cn_result


