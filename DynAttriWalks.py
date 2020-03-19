# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/5 21:31
import torch
import torch.nn
from torch.optim import SGD, Adam
from SkipGramNS import SkipGramNeg
from utils import simulate_walks, DataPipeline, edge_s1_minus_s0, count_word
from utils import unique_nodes_from_edge_set, egde_weight_changed, save_any_obj_pkl
import numpy as np
import networkx as nx
from AutoEncoder import AutoEncoder
import gensim
import pickle

class DynAttriWalks(object):
    def __init__(self, G_dynamic, limit, local_global, num_walks, walk_length, window,
                    emb_dim, n_negative, seed=2019):
        
        self.G_dynamic = G_dynamic.copy()  # 动态graph
        self.limit = limit
        self.local_global = local_global
        self.emb_dim = emb_dim  # 节点embedding的维度
        self.num_walks = num_walks  # 每个节点随机游走的次数
        self.walk_length = walk_length  # 每次随机游走的长度
        self.window = window  # Skip-Gram parameter
        self.n_negative = n_negative  # 负采样的个数
        self.seed = seed  # 随机种子
        self.emb_dicts = []
        self.reservoir = {}
        self.workers = 10

        # if self.cuda:
        #     self.model: SkipGramNeg = SkipGramNeg(vocabulary_size, embedding_size).cuda()  # SGNS模型
        # else:
        #     self.model: SkipGramNeg = SkipGramNeg(vocabulary_size, embedding_size)
        # self.model_optim = SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        w2v = gensim.models.Word2Vec(sentences=None, size=self.emb_dim, window=self.window, sg=1, hs=0,
                                     negative=self.n_negative, ns_exponent=0.75,
                                     alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4,
                                     workers=self.workers, seed=self.seed,
                                     corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                                     max_vocab_size=None, max_final_vocab=None, trim_rule=None)
        for t in range(len(self.G_dynamic)):
            if t == 0:
                G0 = self.G_dynamic[t]
                sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, weighted=True, walk_length=self.walk_length)
                sentences = [[str(j) for j in i] for i in sentences]

                print("-start node embedding on Graph 0" + "/" + str(len(self.G_dynamic)))
                w2v.build_vocab(sentences=sentences, update=False)  # init traning, so update False
                # 利用Word2Vec模型进行训练
                w2v.train(sentences=sentences, total_examples=w2v.corpus_count,
                          epochs=w2v.iter)  # follow w2v constructor
                print("-end node embedding on Graph 0" + "/" + str(len(self.G_dynamic)))

                emb_dict = {}  # {nodeID: emb_vector, ...}
                for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
                save_any_obj_pkl(obj=emb_dict, path="output/collaborate_network(2G)/DynAttriWalks/collaborate_network_" + str(t) + "_embs.pkl")
                self.emb_dicts.append(emb_dict)
            else:
                G0 = self.G_dynamic[t - 1]  # previous graph 之前的graph
                G1 = self.G_dynamic[t]  # current graph 现在的graph
                print("-start selecting nodes on Graph " + str(t) + "/" + str(len(self.G_dynamic)))
                node_update_list, self.reservoir, node_del, node_add = node_selecting_scheme(graph_t0 = G0,
                                                                         graph_t1=G1,
                                                                         reservoir_dict=self.reservoir,
                                                                         limit=self.limit,
                                                                         local_global=self.local_global)
                print("-end selecting nodes on Graph " + str(t) + "/" + str(len(self.G_dynamic)))

                sentences = simulate_walks(nx_graph=G1, num_walks=self.num_walks, weighted=True, walk_length=self.walk_length,
                                           selected_nodes=node_update_list)
                sentences = [[str(j) for j in i] for i in sentences]

                print("-start node embedding on Graph " + str(t) + "/" + str(len(self.G_dynamic)))
                w2v.build_vocab(sentences=sentences, update=True)  # online update
                # 利用Word2Vec模型进行训练
                w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)
                print("-end node embedding on Graph " + str(t) + "/" + str(len(self.G_dynamic)))

                emb_dict = {}  # {nodeID: emb_vector, ...}
                for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
                save_any_obj_pkl(obj=emb_dict, path="output/collaborate_network(2G)/DynAttriWalks/collaborate_network_" + str(t) + "_embs.pkl")
                self.emb_dicts.append(emb_dict)
        return self.emb_dicts

    def save_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
        ''' save # emb_dict @ t0, t1, ... to a file using pickle
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)




def node_selecting_scheme(graph_t0, graph_t1, reservoir_dict, limit=0.1, local_global=0.5):
    ''' select nodes to be updated 选择要更新的节点
         G0: previous graph @ t-1; 前一时刻t-1的graph G0
         G1: current graph  @ t; 当前时刻t的graph G1
         reservoir_dict: will be always maintained in ROM 不断维护的字典
         limit: fix the number of node --> the percentage of nodes of a network to be updated (exclude new nodes)
                除新节点外要更新节点的数量
         local_global: # of nodes from recent changes v.s. from random nodes
                       局部感知与全局拓扑的均衡
    '''
    G0 = graph_t0.copy()
    G1 = graph_t1.copy()
    # 增加的边
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()),
                                s0=set(G0.edges()))  # one may directly use streaming added edges if possible
    # 删除的边
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()),
                                s0=set(G1.edges()))  # one may directly use streaming added edges if possible
    # 权重发生改变的边
    edge_wei, common_edge = egde_weight_changed(G1=G1, G0=G0)

    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)  # unique 增加的边中所有的节点
    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)  # unique 删除的边中所有的节点
    node_affected_by_edge_wei = unique_nodes_from_edge_set(edge_wei)  # unique 删除的边中所有的节点
    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del + node_affected_by_edge_wei))  # unique 所有受影响的节点
    node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]  # 增加的节点
    node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]  # 删除的节点
    # 从reservoir中删除消失的节点
    if len(node_del) != 0:  # 删除的节点不为0，即有消失的节点
        reservoir_key_list = list(reservoir_dict.keys())  # reservoir中的keys
        for node in node_del:
            if node in reservoir_key_list:
                del reservoir_dict[node]  # if node being deleted, also delete it from reservoir
    # affected的节点既在G0中，又在G1中
    exist_node_affected = list(set(node_affected) - set(node_add) - set(node_del))  # affected nodes are in both G0 and G1

    attri_change = {}
    for node in exist_node_affected:
        attri_change[node] = np.linalg.norm(np.array(G0.nodes[node]["attribute"]) - np.array(G1.nodes[node]["attribute"]), ord=2)

    num_limit = int(G1.number_of_nodes() * limit)  # 要更新节点的数量
    local_limit = int(local_global * num_limit)  # 局部感知节点的数量
    global_limit = num_limit - local_limit  # 全局拓扑节点的数量

    node_update_list = []  # all the nodes to be updated 要更新节点的list
    # 选择 最受影响的节点
    most_affected_nodes, reservoir_dict = select_most_affected_nodes(G0, G1, attri_change, local_limit, reservoir_dict,
                                                                     exist_node_affected)
    # 当有变化的节点少于 local_limit节点数量时，随机采样节点用作补偿
    lack = local_limit - len(
        most_affected_nodes)  # if the changes are relatively smaller than local_limit, sample some random nodes for compensation
    # tabu节点为新增节点和最受影响节点的并集
    tabu_nodes = set(node_add + most_affected_nodes)
    # 除tabu节点之外的其他节点
    other_nodes = list(set(G1.nodes()) - tabu_nodes)
    # 从other_nodes中随机选择节点
    random_nodes = list(np.random.choice(other_nodes, min(global_limit + lack, len(other_nodes)), replace=False))
    # 待更新embedding的节点list
    node_update_list = node_add + most_affected_nodes + random_nodes

    reservoir_key_list = list(reservoir_dict.keys())
    node_update_set = set(node_update_list)  # remove repeated nodes due to resample 出去重复节点
    # 已选则某个节点之后，从reservoir中删除，则下次重新开始累积该节点的变化
    for node in node_update_set:
        if node in reservoir_key_list:
            del reservoir_dict[node]  # if updated, delete it

    print(
        f'num_limit {num_limit}, local_limit {local_limit}, global_limit {global_limit}, # nodes updated {len(node_update_list)}')
    print(f'# nodes added {len(node_add)}, # nodes deleted {len(node_del)}')
    print(
        f'# nodes affected {len(node_affected)}, # nodes most affected {len(most_affected_nodes)}, # of random nodes {len(random_nodes)}')
    print(f'num of nodes in reservoir with accumulated changes but not updated {len(list(reservoir_dict))}')
    return node_update_list, reservoir_dict, node_del, node_add


def select_most_affected_nodes(G0, G1, attri_change, num_limit_return_nodes, reservoir_dict, exist_node_affected):
    ''' return num_limit_return_nodes to be updated
         based on the ranking of the accumulated changes w.r.t. their local connectivity
    '''
    most_affected_nodes = []  # 影响最大的节点
    for node in exist_node_affected:  # 遍历受影响的节点
        topology_change = abs(nx.degree(G1, node, weight="weight") - nx.degree(G0, node, weight="weight")) / nx.degree(G0, node, weight="weight")
        attribute_change = attri_change[node]
        changes = topology_change + attribute_change
        # 将changes存入reservoir中，计算节点的累积变化
        if node in reservoir_dict.keys():
            reservoir_dict[node] += changes  # accumulated changes
        else:
            reservoir_dict[node] = changes  # newly added changes

    if len(exist_node_affected) > num_limit_return_nodes:  # 当要选择的节点数量，小于受影响的节点数量时，要对节点进行筛选
        # worse case O(n) https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html
        # the largest change at num_limit_return_nodes will be returned
        # np.partition(list, k)[k]  为list中的第k大元素
        # 故 cutoff_score 为划分最受影响节点的门限
        cutoff_score = np.partition(list(reservoir_dict.values()), -num_limit_return_nodes, kind='introselect')[
            -num_limit_return_nodes]
        cnt = 0  # 已选择的节点数
        for node in reservoir_dict.keys():  # 遍历reservoir分数的键值
            if reservoir_dict[node] >= cutoff_score:  # fix bug: there might be multiple equal cutoff_score nodes...
                if cnt == num_limit_return_nodes:  # fix bug: we need exactly the number of limit return nodes...
                    break  # 已达到待选择的节点数，直接退出
                most_affected_nodes.append(node)  # 大于门限分数，添加节点
                cnt += 1  # 待选择节点 +1
    else:  # NOTE: len(exist_node_affected) <= num_limit_return_nodes
        # 当要选择的节点数量大于受影响的节点数量时，不需筛选，直接返回
        most_affected_nodes = exist_node_affected
    return most_affected_nodes, reservoir_dict


