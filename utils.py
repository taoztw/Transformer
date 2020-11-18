"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'
import numpy as np
import torch

def seq_padding(X, padding=0):
    """
    对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
    """
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    # （注意这里默认padding id是0，相当于是拿<UNK>来做了padding）
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def subsequent_mask(size):
    """
    deocer层self attention需要使用一个mask矩阵，
    :param size: 句子维度
    :return: 右上角(不含对角线)全为False，左下角全为True的mask矩阵
    """
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


def get_word_dict():
    """
    获取中英，word2idx和idx2word字典
    :return: 各个字典
    """
    import csv
    cn_idx2word = {}
    cn_word2idx = {}
    en_idx2word = {}
    en_word2idx = {}
    with open("data/word_name_dict/cn_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            cn_idx2word[int(l[0])] = l[1]
            cn_word2idx[l[1]] = int(l[0])
    with open("data/word_name_dict/en_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            en_idx2word[int(l[0])] = l[1]
            en_word2idx[l[1]] = int(l[0])

    return cn_idx2word, cn_word2idx, en_idx2word, en_word2idx


def bleu_candidate(sentence):
    "保存预测的翻译结果到文件中"
    from setting import BLEU_CANDIDATE
    with open(BLEU_CANDIDATE,'a+',encoding='utf-8') as f:
        f.write(sentence + '\n')


def bleu_references(read_filename, save_filename):
    """
    保存参考译文到文件中。(中文,文件中未空格切分)
    :param file_name:
    :return:
    """
    writer = open(save_filename,'a+',encoding='utf-8')
    with open(read_filename,'r',encoding="utf-8") as f_read:
        for line in f_read:
            line = line.strip().split('\t')
            sentence_tap = " ".join([w for w in line[1]])
            writer.write(sentence_tap+'\n')
    writer.close()
    print('写入成功')



if __name__ == '__main__':
    read_filename = 'data/dev.txt'
    save_filename = 'data/bleu/references.txt'

    bleu_references(read_filename, save_filename)
