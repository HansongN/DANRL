# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/23 10:39 
from utils import load_any_obj_pkl
import os
import numpy as np
import networkx as nx
import json


def recommende_node(scores, top_k, index_node):
    order = np.argsort(scores)[::-1]
    result = []
    for index in order[1:top_k+1]:
        result.append(index_node[index])
    return result


def recommendation(cn, G1, G2, index2node, top_k):
    tp, fp = 0, 0
    recommende_list = []
    for i in range(len(cn)):
        temp = recommende_node(cn[i], top_k, index2node)
        recommende_list.extend([(index2node[i], j) for j in temp])
    for scholars in recommende_list:
        if scholars[0] in G1.nodes() and scholars[1] in G1.nodes():
            if scholars in G1.edges():
                tp += 1
            else:
                fp += 1
        elif scholars[0] in G2.nodes() and scholars[1] in G2.nodes():
            if scholars in G2.edges():
                tp += 1
            else:
                fp += 1
        # elif scholars[0] in G3.nodes() and scholars[1] in G3.nodes():
        #     if scholars in G2.edges():
        #         tp += 1
        #     else:
        #         fp += 1
    # print(tp, fp)
    # print("recommend_precision_score=", "{:.9f}".format(tp/(tp+fp)))
    return tp / (tp + fp)



G_dynamic0 = load_any_obj_pkl("graph_data/collaborate_network(1G)/collaborate_network_2006_2016.pkl")
G_dynamic = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")

print("计算共同邻居")
cn_list = []
index2node_list = []
for g in G_dynamic:
    nodes = list(g.nodes())
    cn_matrix = np.zeros([len(nodes), len(nodes)])
    index2node = dict()
    for i in range(len(nodes)):
        index2node[i] = nodes[i]
        for j in range(i, len(nodes)):
            cn_matrix[i, j] = len(list(nx.common_neighbors(g, nodes[i], nodes[j])))
            cn_matrix[j, i] = cn_matrix[i, j]
    cn_list.append(cn_matrix)
    index2node_list.append(index2node)

print("计算推荐精度")
precisions = dict()
for top_k in range(1, 11):
    print(top_k)
    score = []
    for t in range(len(cn_list) - 3):
        task = recommendation(cn_list[t], G_dynamic0[t+2], G_dynamic0[t+3], index2node_list[t], top_k)
        score.append(task)
    precisions["top_" + str(top_k)] = np.mean(score)
filepath = "evaluation_result/collaborate_network(2G)/recommendation"
if not os.path.exists(filepath):
    os.makedirs(filepath)
filename = "CommonNeighbor_G_2.txt"
output = open(os.path.join(filepath, filename), "w")
output.write(json.dumps(precisions) + "\n")
output.close()
