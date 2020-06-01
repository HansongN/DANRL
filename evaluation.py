# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/16 10:36
from utils import load_any_obj_pkl, gen_test_edge_wrt_changes
from downstream import lpClassifier, recommendation
import numpy as np
import json, os


def load_OpenNE_Embedding(method, year):
    sid_emb = dict()
    with open(r"output/collaborate_network(2G)/" + method + "/collaborate_network_" + str(year) + "_embs.txt",
              "r") as embeddings:
        embeddings.readline()
        for embedding in embeddings:
            l = embedding.split()
            sid_emb[l[0]] = [float(n) for n in l[1:]]
    embeddings.close()
    return sid_emb


G_dynamic_ori = load_any_obj_pkl("graph_data/collaborate_network(1G)/collaborate_network_2006_2016.pkl")
G_dynamic = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")

method = "DeepWalk"
if method == "DANRL":
    emb_dicts = load_any_obj_pkl("output/collaborate_network(2G)/" + method + "/collaborate_network_2007_2016_embs.pkl")
else:
    emb_dicts = []
    for year in range(2007, 2017):
        emb_dicts.append(load_OpenNE_Embedding(method, year))

print(len(emb_dicts))
avg_score = dict()
for top_k in range(1, 11, 1):
    score = []
    for t in range(len(emb_dicts) - 2):  # 遍历所有time step的embedding
        model = recommendation(emb_dicts[t], G0=G_dynamic[t], G1=G_dynamic_ori[t + 2], G2=G_dynamic_ori[t + 3],
                               # G3=G_dynamic_ori[t+4]
                               )
        score.append(model.evaluate_precision_k(top_k))
    avg_score["top_" + str(top_k)] = np.mean(score)


filepath = "evaluation_result/collaborate_network(2G)/recommendation"
if not os.path.exists(filepath):
    os.makedirs(filepath)
filename = method + "_G_2ori.txt"
output = open(os.path.join(filepath, filename), "w")
output.write(json.dumps(avg_score) + "\n")
output.close()
