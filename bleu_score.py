"""
Created by tz on 2020/11/18
"""

__author__ = 'tz'

from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

# 单个句子
# reference = [['this', 'is', 'small', 'test']]
# candidate = ['this', 'is', 'a', 'test']
# score = sentence_bleu(reference, candidate,weights=(1,0.5,0.5,0.5))

# 多个句子
def read_references():
    """
    预料的refetences计算
    :return: [ [['word','word'],['word','word']]   ]
    """
    result = []
    r_sentences = []

    from setting import BLEU_REFERENCES
    f = open(BLEU_REFERENCES,'r',encoding='utf-8')
    sentences = f.readlines()
    for s in sentences:
        references = []
        references.append(s.strip().split(' '))
        result.append(references)
    f.close()
    return result

def read_candidates():
    result = []
    from setting import BLEU_CANDIDATE
    file = open(BLEU_CANDIDATE,'r',encoding='utf-8')
    sentences = file.readlines()
    for s in sentences:
        result.append(s.strip().split(' '))
    file.close()
    return result

# references = [[['this', 'is', 'small', 'test']]]
# candidates = [['this', 'is', 'a', 'test']]
if __name__ == '__main__':

    references = read_references()
    candidates = read_candidates()
    score = corpus_bleu(references, candidates,weights=(1,0.2,0,0))

    print(score)