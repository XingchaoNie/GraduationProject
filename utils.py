import numpy as np
import torch


def high_low_trans(s):
    # 将传入的标签矩阵进行正则化
    split_standard = 3
    tf = s <= split_standard
    res = np.ones(np.shape(tf))
    r, c = np.shape(res)
    for i in range(r):
        for j in range(c):
            if tf[i, j]:
                res[i, j] = 1
            else:
                res[i, j] = 0
    return res


def label_trans(s, kinds):
    """
    :param s:传入一个标签向量
    :param kinds: 标签的种类数
    :return: 对每个标签都转换为标签向量，最终传回一个标签向量矩阵
    例如对最重要分类的状态有3种，则标签0对应[1, 0, 0], 标签1对应[0, 1, 0], 也就是变成softmax回归
    """
    s = s.reshape((-1))
    label_num = s.shape[0]
    res = torch.zeros((label_num, kinds))
    for i in range(label_num):
        res[i, int(s[i])] = 1
    return res

def res_trans(y):
    res = torch.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        if y[i, 0] > y[i, 1]:
            res[i] = 0
        else:
            res[i] = 1
    return res

