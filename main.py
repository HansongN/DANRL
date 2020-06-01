# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/6 10:49
from utils import load_any_obj_pkl, save_any_obj_pkl
from DANRL import DANRL

ratio_most_affected_nodes = 0.2
num_walks = 20
walk_length = 80
window_size = 10
embedding_dimensional = 128
num_negative = 10

G_dynamic = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")


model = DANRL(G_dynamic=G_dynamic,
                      limit=ratio_most_affected_nodes,
                      local_global=1,
                      num_walks=num_walks,
                      walk_length=walk_length,
                      window=window_size,
                      emb_dim=embedding_dimensional,
                      n_negative=num_negative)
emb_dicts = model.train()
save_any_obj_pkl(obj=emb_dicts, path="output/collaborate_network(2G)/DANRL/collaborate_network_2007_2016_embs.pkl")

# import os
# import numpy as np
# col_net = "collaborate_network(2G)"
# G_dynamic = load_any_obj_pkl("graph_data/"+ col_net + "/collaborate_network_2007_2016.pkl")
# for dim in [16, 64, 128, 256]:
#     for win in range(16, 42, 2):
#         model = DANRL(G_dynamic=G_dynamic,
#                               limit=0.2,
#                               local_global=1,
#                               num_walks=20,
#                               walk_length=80,
#                               window=win,
#                               emb_dim=dim,
#                               n_negative=10)
#         emb_dicts = model.train()
#         filepath = "parameter_sensitivity/"+ col_net + "/output_0526"
#         filename = "2007_2016_embs_window_" + str(win) + "_dim_" + str(dim) + ".pkl"
#         if not os.path.exists(filepath):
#             os.makedirs(filepath)
#         save_any_obj_pkl(obj=emb_dicts, path=os.path.join(filepath, filename))
