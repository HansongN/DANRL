# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/16 10:39 
import networkx as nx
import numpy as np
from utils import pairwise_similarity, cosine_similarity, ranking_precision_score, node_id2idx, auc_score, average_precision_score


class grClassifier(object):
    def __init__(self, emb_dict, rc_graph):
        self.graph = rc_graph
        self.embeddings = emb_dict
        self.adj_mat, self.score_mat = self.gen_test_data_wrt_graph_truth(graph=rc_graph)  # 根据graph生成邻接矩阵和score矩阵

    def gen_test_data_wrt_graph_truth(self, graph):
        ''' input: a networkx graph
            output: adj matrix and score matrix; note both matrices are symmetric
            输入：networkx的graph
            输出：邻接矩阵和score（相似性）矩阵，都是对称的
        '''
        G = graph.copy()  # 复制graph，防止改变原graph
        adj_mat = nx.to_numpy_array(G=G, nodelist=None)  # ordered by G.nodes(); n-by-n  按G.nodes()的顺序生成邻接矩阵

        # 将加权邻接矩阵变为不加权的邻接矩阵
        adj_mat = np.where(adj_mat == 0, 0, 1)  # vectorized implementation weighted -> unweighted if necessary

        emb_mat = []  # graph节点的embedding矩阵
        for node in G.nodes():
            emb_mat.append(self.embeddings[node])
        # 使用节点间embedding的cosine相似性作为score矩阵
        score_mat = pairwise_similarity(emb_mat, type='cosine')  # n-by-n corresponding to adj_mat
        return np.array(adj_mat), np.array(score_mat)

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k; to be merged with average_precision_score()
        使用graph中节点的相似性矩阵作为对原图的重构
        ground truth为原图的邻接矩阵
        '''
        pk_list = []
        if node_list == None:  # eval all nodes  不指定node_list，则对所有node进行评估
            size = self.adj_mat.shape[0]  # num of rows -> num of nodes 节点的个数
            for i in range(size):
                # 计算每个节点的precision，并加入pk_list
                pk_list.append(ranking_precision_score(self.adj_mat[i], self.score_mat[i],
                                                       k=top_k))  # ranking_precision_score 计算每个节点的precision，并加入pk
        else:  # only eval on node_list 只在指定的node上计算precision
            if len(node_list) == 0:  # if there is no testing data (dyn networks not changed), set auc to 1
                # 没有数据，设precision为1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)  # 记录node_list中的节点在graph中的index list
                new_adj_mat = [self.adj_mat[i] for i in node_idx]  # 针对node_list中的节点构建新的邻接矩阵
                new_score_mat = [self.score_mat[i] for i in node_idx]  # 针对node_list中的节点构建新的score矩阵
                size = len(new_adj_mat)  # node_list中节点的个数
                for i in range(size):
                    # 计算node_lsit中每个节点的precision，并加入pk_list
                    pk_list.append(ranking_precision_score(new_adj_mat[i], new_score_mat[i],
                                                           k=top_k))  # ranking_precision_score only on node_list
        print("ranking_precision_score=", "{:.9f}".format(np.mean(pk_list)))

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average precision at rank k; to be merged with evaluate_precision_k()
        '''
        pk_list = []
        if node_list == None:  # eval all nodes
            size = self.adj_mat.shape[0]  # num of rows -> num of nodes
            for i in range(size):
                pk_list.append(
                    average_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k))  # average_precision_score
        else:  # only eval on node_list
            if len(node_list) == 0:  # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list.append(average_precision_score(new_adj_mat[i], new_score_mat[i],
                                                           k=top_k))  # average_precision_score only on node_list
        print("average_precision_score=", "{:.9f}".format(np.mean(pk_list)))


class recommendation(object):
    def __init__(self, emb_dict, G0, G1, G2, G3=None):
        self.G0 = G0
        self.G1 = G1
        self.G2 = G2
        self.G3 = G3
        self.embeddings = emb_dict
        self.score_mat, self.index_node = self.gen_test_data_wrt_graph_truth(graph=G0)  # 根据graph生成邻接矩阵和score矩阵

    def gen_test_data_wrt_graph_truth(self, graph):
        ''' input: a networkx graph
            output: score matrix; symmetric
            输入：networkx的graph
            输出：score（相似性）矩阵, 对称的
        '''
        G = graph.copy()  # 复制graph，防止改变原graph
        emb_mat = []  # graph节点的embedding矩阵
        index_node = {}
        index = 0
        for node in G.nodes():
            index_node[index] = node
            index += 1
            emb_mat.append(self.embeddings[node])
        # 使用节点间embedding的cosine相似性作为score矩阵
        score_mat = pairwise_similarity(emb_mat, type='cosine')  # n-by-n corresponding to adj_mat
        return np.array(score_mat), index_node

    def evaluate_precision_k(self, top_k):
        ''' Precision at rank k; to be merged with average_precision_score()
        使用graph中节点的相似性矩阵作为对原图的重构
        ground truth为原图的邻接矩阵
        '''
        tp, fp = 0, 0
        recommende_list = []
        from utils import recommende_node
        for i in range(len(self.score_mat)):
            temp = recommende_node(self.score_mat[i], top_k, self.index_node)
            recommende_list.extend([(self.index_node[i], j) for j in temp])
        for scholars in recommende_list:
            if scholars[0] in self.G1.nodes() and scholars[1] in self.G1.nodes():
                if scholars in self.G1.edges():
                    tp += 1
                else:
                    fp += 1
            elif scholars[0] in self.G2.nodes() and scholars[1] in self.G2.nodes():
                if scholars in self.G2.edges():
                    tp += 1
                else:
                    fp += 1
            # elif scholars[0] in self.G3.nodes() and scholars[1] in self.G3.nodes():
            #     if scholars in self.G2.edges():
            #         tp += 1
            #     else:
            #         fp += 1
        # print(tp, fp)
        # print("recommend_precision_score=", "{:.9f}".format(tp/(tp+fp)))
        return tp/(tp+fp)


class lpClassifier(object):
    def __init__(self, emb_dict):
        self.embeddings = emb_dict

    # clf here is simply a similarity/distance metric
    def evaluate_auc(self, X_test, Y_test):
        test_size = len(X_test)
        Y_true = [int(i) for i in Y_test]
        Y_probs = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[X_test[i][0]]).reshape(-1, 1)
            end_node_emb = np.array(
                self.embeddings[X_test[i][1]]).reshape(-1, 1)
            # ranging from [-1, +1]
            score = cosine_similarity(start_node_emb, end_node_emb)
            # switch to prob... however, we may also directly y_score = score
            Y_probs.append((score + 1) / 2.0)
            # Y_probs.append(score)
            # in sklearn roc... which yields the same reasult
        if len(Y_true) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            auc = auc_score(y_true=Y_true, y_score=Y_probs)
        print("auc=", "{:.9f}".format(auc))
