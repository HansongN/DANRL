# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/6 10:09 
import time
import random
import collections
import numpy as np
import pickle
import networkx as nx

def average_precision_score(y_true, y_score, k=10):
    """ Average precision at rank k
        y_true & y_score; array-like, shape = [n_samples]
        see https://gist.github.com/mblondel/7337391
    """
    unique_y = np.unique(y_true)
    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    pos_label = unique_y[1] # 1 as true
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1][:min(n_pos, k)] # note: if k>n_pos, we use fixed n_pos; otherwise use given k
    y_pred_true = np.asarray(y_true)[order]
    score = 0
    for i in range(len(y_pred_true)):
        if y_pred_true[i] == pos_label: # if pred_true == ground truth positive label
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(i + 1):  # precision @1, @2, ..., @ min(n_pos, k)
                if y_pred_true[j] == pos_label: # pred true --> ground truth also positive
                    prec += 1.0
            prec /= (i + 1.0)  # precision @i where i=1,2, ... ; note: i+1.0 since i start from 0
            score += prec
    if n_pos == 0:
        return 0
    return score / n_pos # micro-score; if macro-score use np.sum(score)/np.size(score)


def simulate_walks(nx_graph, num_walks, walk_length, weighted=False, restart_prob=None, selected_nodes=None):
    '''
    Repeatedly simulate random walks from each node
    在nx_graph的每个节点上执行随机游走
    '''
    G = nx_graph  # graph
    walks = []  # 游走的list

    if selected_nodes == None:  # 在所有节点上执行所有游走
        nodes = list(G.nodes())
    else:  # 在选定的节点上执行随机游走
        nodes = list(selected_nodes)

    ''' multi-processors; use it iff the # of nodes over 20k  # 节点数量超过20k时，多进程执行随机游走
    if restart_prob == None: # naive random walk
         t1 = time.time()
         for walk_iter in range(num_walks):
              random.shuffle(nodes)
              from itertools import repeat
              from multiprocessing import Pool, freeze_support
              with Pool(processes=5) as pool:
                   # results = [pool.apply_async(random_walk, args=(G, node, walk_length)) for node in nodes]
                   # results = [p.get() for p in results]
                   results = pool.starmap(random_walk, zip(repeat(G), nodes, repeat(walk_length)))
              for result in results:
                   walks.append(result)
         t2 = time.time()
         print('all walks',len(walks))
         print(f'random walk sampling, time cost: {(t2-t1):.2f}')
    '''

    if restart_prob == None:  # naive random walk  不带重启的随机游走
        t1 = time.time()
        if weighted == False:
            for walk_iter in range(num_walks):  # num_walks: 每个节点执行的随机游走
                random.shuffle(nodes)  # 随机排序所有的节点
                for node in nodes:
                    # 在每个节点上执行随机游走，并加入walks
                    walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
        else:
            for walk_iter in range(num_walks):  # num_walks: 每个节点执行的随机游走
                random.shuffle(nodes)  # 随机排序所有的节点
                for node in nodes:
                    # 在每个节点上执行随机游走，并加入walks
                    walks.append(random_walk_weight(nx_graph=G, start_node=node, walk_length=walk_length))
        t2 = time.time()
        print(f'random walk sampling, time cost: {(t2 - t1):.2f}')
    else:  # random walk with restart 带重启的随机游走
        t1 = time.time()
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length,
                                                 restart_prob=restart_prob))
        t2 = time.time()
        print(f'-random walk sampling, time cost: {(t2 - t1):.2f}')
    return walks


def random_walk(nx_graph, start_node, walk_length):
    '''
    Simulate a random walk starting from start node
    模拟开始于start node的随机游走
    '''
    G = nx_graph
    walk = [start_node]  # 随机游走的list，首个节点为start_node

    while len(walk) < walk_length:  # 当目前游走长度小于规定的游走长度时，继续循环
        cur = walk[-1]  # 当前节点为游走的最后一个节点
        cur_nbrs = list(G.neighbors(cur))  # 当前节点的邻域
        if len(cur_nbrs) > 0:  # 当邻域中节点数量大于0
            walk.append(random.choice(cur_nbrs))  # 从邻域中随机选择一个节点，游走到该节点，加入walk
        else:  # 当邻域中节点数量等于0
            break
    return walk


def random_walk_weight(nx_graph, start_node, walk_length):
    '''
    Simulate a random walk starting from start node
    模拟开始于start node的随机游走
    '''
    G = nx_graph
    walk = [start_node]  # 随机游走的list，首个节点为start_node

    while len(walk) < walk_length:  # 当目前游走长度小于规定的游走长度时，继续循环
        cur = walk[-1]  # 当前节点为游走的最后一个节点
        cur_nbrs = list(G.neighbors(cur))  # 当前节点的邻域
        unigram_table = []
        for cur_nbr in cur_nbrs:
            w = G.get_edge_data(cur, cur_nbr)["weight"]
            unigram_table.extend([cur_nbr] * int(w * 10))
        if len(cur_nbrs) > 0:  # 当邻域中节点数量大于0
            walk.append(random.choice(unigram_table))  # 从邻域中随机选择一个节点，游走到该节点，加入walk
        else:  # 当邻域中节点数量等于0
            break
    return walk


def random_walk_restart(nx_graph, start_node, walk_length, restart_prob):
    '''
    random walk with restart
    restart if p < restart_prob
    带重启的随机游走，p小于restart_prob时，返回start_node
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        p = random.uniform(0, 1)  # (0, 1) 中随机采样
        if p < restart_prob:  # 重启条件
            cur = walk[0]  # restart
            walk.append(cur)
        else:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
    return walk


def noise(vocabs, word_count, node2index):
    """
    generate noise distribution
    :param vocabs:  词汇表
    :param word_count:  [ [word_index, word_count], ]
    :return:
    """
    Z = 0.001
    unigram_table = []
    temp = []
    word2index = dict()
    index = 0
    for word, count in word_count.items():
        word2index[word] = index
        index += 1
        temp.append([word, count])
    word_count = temp
    num_total_words = sum([c for w, c in word_count])  # 总的单词数
    for vo in vocabs:  # 词汇表中每个单词
        unigram_table.extend([node2index[vo]] * int(((word_count[word2index[vo]][1]/num_total_words)**0.75)/Z))
    print("vocabulary size", len(vocabs))
    print("unigram_table size:", len(unigram_table))
    return unigram_table


class DataPipeline:
    """
    - data: [word_index]
    - vocabs: list(set(data))
    - word_count: [ [word_index, word_count], ]
    - data_index: data_offest
    """
    def __init__(self, data, vocabs, node2index, word_count, walk_length, data_index=0, row_index=0, use_noise_neg=True):
        self.data = data
        self.data_index = data_index
        self.row_index = row_index
        self.walk_length = walk_length
        if use_noise_neg:  # 是否进行负采样
            self.unigram_table = noise(vocabs, word_count, node2index)
        else:
            self.unigram_table = vocabs

    def get_neg_data(self, batch_size, num, target_inputs):
        """
        sample the negative data. Don't use np.random.choice(), it is very slow.
        :param batch_size: int
        :param num: int
        :param target_inputs: []
        :return:
        """
        neg = np.zeros((num))
        for i in range(batch_size):
            delta = random.sample(self.unigram_table, num)
            while target_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

    def generate_batch(self, batch_size, num_skips, skip_window):
        """
        get the data batch
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return: target batch and context batch
        """
        num_windows_per_sentence = self.walk_length - 2 * skip_window  # 60
        num_words_per_sentence = num_windows_per_sentence * num_skips
        assert batch_size % num_words_per_sentence == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window, target, skip_window ] 3
        for i in range(batch_size // num_words_per_sentence):
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(self.data[self.row_index][self.data_index])
                self.data_index = (self.data_index + 1) % self.walk_length
            for j in range(num_windows_per_sentence):
                for k in range(skip_window):
                    batch[i * num_words_per_sentence + j * num_skips + 2 * k] = buffer[skip_window]
                    batch[i * num_words_per_sentence + j * num_skips + 2 * k + 1] = buffer[skip_window]
                    labels[i * num_words_per_sentence + j * num_skips + 2 * k] = buffer[skip_window - k - 1]
                    labels[i * num_words_per_sentence + j * num_skips + 2 * k + 1] = buffer[skip_window + k + 1]
                buffer.append(self.data[self.row_index][self.data_index])
                self.data_index = (self.data_index + 1) % self.walk_length
            self.row_index = (self.row_index + 1) % len(self.data)
            self.data_index = 0
        # for _ in range(span):
        #     buffer.append(self.data[self.row_index][self.data_index])  # 滑动窗口内的单词  buffer.append(data[0]) buffer.append(data[1]) buffer.append(data[2])
        #     self.data_index = (self.data_index + 1) % len(self.data)  # data_index加1
        # for i in range(batch_size // num_skips):  # 64: 0, 1
        #     target = skip_window  # 1
        #     targets_to_avoid = [skip_window]  # [1]
        #     for j in range(num_skips):  # 2: 0, 1
        #         while target in targets_to_avoid:
        #             target = random.randint(0, span - 1)  # [0, 2] 0, 2
        #         targets_to_avoid.append(target)  # [1, 0, 2]
        #         batch[i * num_skips + j] = buffer[skip_window]  # batch[0] = buffer[1]  batch[1] = buffer[1]
        #         labels[i * num_skips + j] = buffer[target]  # label[0] = buffer[0]  batch[1] = buffer[2]
        #     buffer.append(self.data[self.data_index])  # buffer.append(data[3])
        #     self.data_index = (self.data_index + 1) % len(self.data)  # data_index加1
        # self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels


def edge_s1_minus_s0(s1, s0, is_directed=False):
    ''' s1 and s0: edge/node-pairs set
    '''
    # 当前只支持无向边
    if not is_directed:
        s1_reordered = set( (a,b) if a<b else (b,a) for a,b in s1 )  # 对二者重新排序
        s0_reordered = set( (a,b) if a<b else (b,a) for a,b in s0 )
        return s1_reordered-s0_reordered  # 找到存在于s1，而不存在于s0中的边
    else:
        print('currently not support directed case')


def egde_weight_changed(G1, G0):
    common_edges = G0.edges() & G1.edges()
    weight_change_edge = []
    for edge in common_edges:
        edge = list(edge)
        if G0.get_edge_data(edge[0], edge[1])["weight"] != G1.get_edge_data(edge[0], edge[1])["weight"]:
            weight_change_edge.append(tuple(edge))
    return weight_change_edge, common_edges


def unique_nodes_from_edge_set(edge_set):
    ''' take out unique nodes from edge set
    '''
    """
    不重复的取出 边集 edge_set中的所有节点
    """
    unique_nodes = []
    for a, b in edge_set:
        if a not in unique_nodes:
            unique_nodes.append(a)
        if b not in unique_nodes:
            unique_nodes.append(b)
    return unique_nodes


def count_word(sentence):
    d = dict(collections.Counter(sentence))
    # count = []
    # for key, value in d.items():
    #     count.append([key, value])
    return d


def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open(path, 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def pairwise_similarity(mat, type='cosine'):
    ''' pairwise similarity; can be used as score function;
        vectorized computation
        节点间的成对相似性用于score函数
        默认相似性类型为cosine
    '''
    if type == 'cosine':  # support sprase and dense mat
        from sklearn.metrics.pairwise import cosine_similarity
        result = cosine_similarity(mat, dense_output=True)
    elif type == 'jaccard':
        from sklearn.metrics import jaccard_similarity_score
        from sklearn.metrics.pairwise import pairwise_distances
        # n_jobs=-1 means using all CPU for parallel computing
        result = pairwise_distances(mat.todense(), metric=jaccard_similarity_score, n_jobs=-1)
    elif type == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        # note: similarity = - distance
        result = euclidean_distances(mat)
        result = -result
    elif type == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        # note: similarity = - distance
        result = manhattan_distances(mat)
        result = -result
    else:
        print('Please choose from: cosine, jaccard, euclidean or manhattan')
        return 'Not found!'
    return result


def cosine_similarity(a, b):
    # 计算两个向量间的余弦相似度
    from numpy import dot
    from numpy.linalg import norm
    ''' cosine similarity; can be used as score function; vector by vector; 
        If consider similarity for all pairs,
        pairwise_similarity() implementation may be more efficient
    '''
    a = np.reshape(a,-1)
    b = np.reshape(b,-1)
    if norm(a)*norm(b) == 0:
        return 0.0
    else:
        return dot(a, b)/(norm(a)*norm(b))


def ranking_precision_score(y_true, y_score, k=10):
    """ Precision at rank k
        y_true & y_score; array-like, shape = [n_samples]
        see https://gist.github.com/mblondel/7337391
    """
    unique_y = np.unique(y_true)  # ground truth中label的种类，只支持对两个label的数据计算precision
    if len(unique_y) > 2:  # label种类大于2时，报错
        raise ValueError("Only supported for two relevance levels.")
    pos_label = unique_y[1] # 1 as true 将unique_y中的第二种label看作 正label
    order = np.argsort(y_score)[::-1] # return index 对y_score的index按对应值从大到小排列（原为从小到大，[::-1]为倒序），将这些index对应的节点对看作有边，即邻接矩阵中的值为1
    y_pred_true = np.take(y_true, order[:k]) # predict to be true @k  按order的前k个index从y_true中取值
    n_relevant = np.sum(y_pred_true == pos_label) # predict to be true @k but how many are correct  计算判断正确的有多少个
    # Divide by min(n_pos, k) such that the best achievable score is always 1.0 (note: if k>n_pos, we use fixed n_pos; otherwise use given k)
    #用min（n_pos，k）除以使可获得的最佳分数始终为1.0（注意：如果k > n_pos，我们使用固定的n_pos；否则使用给定的k）
    n_pos = np.sum(y_true == pos_label)  # y_true中正标签的个数
    return float(n_relevant) / min(n_pos, k)
    # return float(n_relevant) / k # this is also fair but can not always get 1.0


def node_id2idx(graph, node_id):
    # 返回graph中指定节点list的index list
    G = graph
    all_nodes = list(G.nodes())
    node_idx = []
    for node in node_id:
        node_idx.append(all_nodes.index(node))
    return node_idx


def auc_score(y_true, y_score):
    ''' use sklearn roc_auc_score API
        y_true & y_score; array-like, shape = [n_samples]
    '''
    from sklearn.metrics import roc_auc_score
    roc = roc_auc_score(y_true=y_true, y_score=y_score)  # 直接调用sklearn中的函数计算roc_auc
    if roc < 0.5:
        # 这是一个二值的分类器，当roc小于0.5时，视为预测二值（0与1）中的另一个值，则要 1 - 0.5
        roc = 1.0 - roc  # since binary clf, just predict the opposite if<0.5
    return roc


def gen_test_edge_wrt_changes(graph_t0, graph_t1):
    ''' input: two networkx graphs
        generate **changed** testing edges for link prediction task
        currently, we only consider pos_neg_ratio = 1.0
        return: pos_edges_with_label [(node1, node2, 1), (), ...]
                neg_edges_with_label [(node3, node4, 0), (), ...]
    '''
    G0 = graph_t0.copy()
    G1 = graph_t1.copy()  # use copy to avoid problem caused by G1.remove_node(node)
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
    unseen_nodes = set(G1.nodes()) - set(G0.nodes())
    delete_nodes = set(G0.nodes()) - set(G1.nodes())
    for node in unseen_nodes:  # to avoid unseen nodes while testing
        G1.remove_node(node)

    edge_add_unseen_node = []  # to avoid unseen nodes while testing
    for node in unseen_nodes:
        for edge in edge_add:
            if node in edge:
                edge_add_unseen_node.append(edge)
    edge_add = edge_add - set(edge_add_unseen_node)
    edge_del_delete_node = []  # to avoid unseen nodes while testing
    for node in delete_nodes:
        for edge in edge_del:
            if node in edge:
                edge_del_delete_node.append(edge)
    edge_del = edge_del - set(edge_del_delete_node)
    pos_edges_with_label = [list(item + (1,)) for item in edge_add]
    neg_edges_with_label = [list(item + (0,)) for item in edge_del]
    if len(edge_add) > len(edge_del):
        num = len(edge_add) - len(edge_del)
        i = 0
        for non_edge in nx.non_edges(G1):
            if non_edge not in edge_del:
                neg_edges_with_label.append(list(non_edge + (0,)))
                i += 1
            if i >= num:
                break
    elif len(edge_add) < len(edge_del):
        num = len(edge_del) - len(edge_add)
        i = 0
        for edge in nx.edges(G1):
            if edge not in edge_add:
                pos_edges_with_label.append(list(edge + (1,)))
                i += 1
            if i >= num:
                break
    else:  # len(edge_add) == len(edge_del)
        pass
    return pos_edges_with_label, neg_edges_with_label


def recommende_node(scores, top_k, index_node):
    order = np.argsort(scores)[::-1]
    result = []
    for index in order[1:top_k+1]:
        result.append(index_node[index])
    return result


