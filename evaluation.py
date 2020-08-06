# -*- coding: utf-8 -*-
# @Author: hliu.Luke
# @Date:   2020-07-31 11:18:22
# @Last Modified by:   hliu.Luke
# @Last Modified time: 2020-08-03 15:17:32
import math
import torch
import numpy as np

# efficient version
def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)
    #print(hits)

    r = np.array(rankedlist)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)
    #print(float(count / k))
    #print(float(count / len(test_matrix)))
    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)

def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c+1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)


def mse_rmse_mae(pred, rate):
	mse  = torch.mean(torch.pow(pred-rate, 2.0))
	rmse = torch.sqrt(mse)
	mae  = torch.mean(torch.abs(pred-rate))
	return mse, rmse, mae