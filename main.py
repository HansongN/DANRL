# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/6 10:49
from utils import load_any_obj_pkl, save_any_obj_pkl
from DynAttriWalks import DynAttriWalks

G_dynamic = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")


model = DynAttriWalks(G_dynamic=G_dynamic, limit=0.2, local_global=1, num_walks=20, walk_length=80, window=10, emb_dim=128, n_negative=10)
emb_dicts = model.train()
save_any_obj_pkl(obj=emb_dicts, path="output/collaborate_network(2G)/DynAttriWalks/collaborate_network_2007_2016_embs.pkl")

# import os
# import numpy as np
# col_net = "collaborate_network(2G)"
# G_dynamic = load_any_obj_pkl("graph_data/"+ col_net + "/collaborate_network_2007_2016.pkl")
# for win in np.arange(18, 22, 2):
#     for dim in [16, 32, 64, 128, 256, 512]:
#         print(win, dim)
#         model = DynAttriWalks(G_dynamic=G_dynamic, limit=0.2, local_global=1, num_walks=20, walk_length=80, window=win, emb_dim=dim, n_negative=10)
#         emb_dicts = model.train()
#         filepath = "parameter_sensitivity/"+ col_net + "/output"
#         filename = "2007_2016_embs_window_" + str(win) + "_dim_" + str(int(dim)) + ".pkl"
#         if not os.path.exists(filepath):
#             os.makedirs(filepath)
#         save_any_obj_pkl(obj=emb_dicts, path=os.path.join(filepath, filename))

# import os
# import numpy as np
# col_net = "collaborate_network(2G)"
# G_dynamic = load_any_obj_pkl("graph_data/"+ col_net + "/collaborate_network_2007_2016.pkl")
# model = DynAttriWalks(G_dynamic=G_dynamic, limit=0.2, local_global=1, num_walks=20, walk_length=80, window=10, emb_dim=2, n_negative=10)
# emb_dicts = model.train()
# filename = "2007_2016_embs_dim_2.pkl"
# save_any_obj_pkl(obj=emb_dicts, path=filename)
