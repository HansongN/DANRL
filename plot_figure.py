# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/18 20:40
import matplotlib.pyplot as plt
import numpy as np
import os
import json

filepath = "evaluation_result/collaborate_network(2G)/recommendation"
methods = ["DynAttriWalks", "DeepWalk", "TADW", "CommonNeighbor"]
colors = ["r", "b", "m", "g"]
markers = ["o", "x", "D", "^"]
score = []
for method in methods:
    filename = method + "_G_2ori.txt"
    with open(os.path.join(filepath, filename), "r") as lines:
        for line in lines:
            line_json = json.loads(line)
            score.append(list(line_json.values()))
# m_s = dict()
# i = 0
# # for m in method:
# #     m_s[m] = score[10 * i : 10*i + 10]
# #     i += 1
for i in range(len(score)):
    plt.plot(np.arange(10), score[i], label=methods[i], color=colors[i], marker=markers[i])
plt.legend(loc="upper right")
plt.title("auc")
plt.show()
