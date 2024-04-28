import os.path
from collections import Counter
import math
from tqdm import tqdm
import pickle

"""
定义计算tfidf公式的函数
"""


def tf(word, count):
    """
    word可以通过count得到，count可以通过countlist得到
    count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
    """
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    """
    统计含有该单词的句子数
    """
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    """
    len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
    """
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def idf_dict(count_list, idf_save_dir):
    """
    另一种方式，预先计算所有的word的idf值
    """
    if idf_save_dir is not None and os.path.exists(idf_save_dir):
        result = pickle.load(open(idf_save_dir, 'rb'))
        print(f'Load idf dict from {idf_save_dir}.')
    else:
        result = {}
        for count in tqdm(count_list):
            for word in count:  # count中不会出现重复word，所以不会出现一个word出现两次而加二的情况
                if word not in result:  # 第一次出现
                    result[word] = 1
                else:
                    result[word] += 1  # 包含该word的doc数目加+1
        for (word, num_contain) in result.items():
            result[word] = math.log(len(count_list) / (1 + num_contain))
        result['_DEFAULT_'] = math.log(len(count_list) / 1)  # 没见过的词赋予一个默认值
        if idf_save_dir is not None:
            pickle.dump(result, open(idf_save_dir, 'wb'))
            print(f'Saved idf dict at {idf_save_dir}.')
    return result


def tfidf(word, count, idf_dict):  # 计算单个样本中word的tfidf
    """
    将tf和idf相乘
    """
    return round(tf(word, count) * (idf_dict[word] if word in idf_dict else idf_dict['_DEFAULT_']), 5)


# word_list sample
"""
[['this', 'is', 'the', 'first', 'document'],
 ['this', 'is', 'the', 'second', 'second', 'document'],
 ['and', 'the', 'third', 'one'],
 ['is', 'this', 'the', 'first', 'document']]
"""


# doc_list用于计算idf，text_list是用于计算tfidf的语料，返回的也是text_list对应的结果
def TF_IDF(text_list, doc_list=None, idf_save_dir=None):
    # 统计词频
    text_countlist = []
    for i in range(len(text_list)):
        count = Counter(text_list[i])
        text_countlist.append(count)
    doc_countlist = []
    if doc_list is not None:
        for i in range(len(doc_list)):
            count = Counter(doc_list[i])
            doc_countlist.append(count)
    else:
        doc_countlist = text_countlist
    # 计算每个单词的tfidf值
    # 预先计算idf值
    idf_dict_ = idf_dict(doc_countlist, idf_save_dir)
    scores_list = []
    for i, count in enumerate(tqdm(text_countlist)):
        scores = {word: tfidf(word, count, idf_dict=idf_dict_) for word in count}
        scores_list.append(scores)
    assert len(scores_list) == len(text_list)
    return scores_list
