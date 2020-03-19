# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2020/02/07 16:06
from utils import load_any_obj_pkl
from downstream import  recommendation
import numpy as np
import os, json
import matplotlib.pyplot as plt

def evaluation():
    G_dynamic_ori = load_any_obj_pkl("graph_data/collaborate_network(1G)/collaborate_network_2006_2016.pkl")
    G_dynamic = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")
    method = "DynAttriWalks"
    filepath = "parameter_sensitivity/collaborate_network(2G)/output"
    files = os.listdir(filepath)

    filepath0 = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    files0 = os.listdir(filepath0)

    i = 0
    for file in files:
        print(i, len(files))
        i += 1
        file_1 = file[:-4] + "_G_2ori.txt"
        if file_1 not in files0:
            emb_dicts = load_any_obj_pkl(os.path.join(filepath, file))
            # print(len(emb_dicts))
            avg_score = dict()
            for top_k in range(1, 11, 1):
                score = []
                for t in range(len(emb_dicts) - 2):  # 遍历所有time step的embedding
                    model = recommendation(emb_dicts[t],
                                           G0=G_dynamic[t],
                                           G1=G_dynamic_ori[t + 2],
                                           G2=G_dynamic_ori[t + 3],
                                           # G3=G_dynamic_ori[t+4]
                                           )
                    score.append(model.evaluate_precision_k(top_k))
                avg_score["top_" + str(top_k)] = np.mean(score)

            # "parameter_sensitivity/collaborate_network(2G)/output"
            output_filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)
            output = open(os.path.join(output_filepath, file[:-4] + "_G_2ori.txt"), "w")
            output.write(json.dumps(avg_score) + "\n")
            output.close()


def limit_para():
    limit = np.arange(0.1, 1, 0.1)
    neg = np.arange(5, 35, 5)

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in limit:
        temp = []
        for p1 in neg:
            filename = "2007_2016_embs_limit_"+str(round(p0, 1))+"_neg_"+str(int(p1))+"_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T
    index = 0
    fig = plt.figure()
    plt.plot([round(n, 1) for n in limit], score[0], label="$n$=" + str(neg[0]), color=colors[index], marker=markers[index])
    index += 1
    for i in range(3):
        plt.plot([round(n, 1) for n in limit], score[i*2+1], label="$n$="+str(neg[i*2+1]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$m$", fontdict={"size": label_size}) # label_size tick_size legend_size
    plt.xticks(fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    plt.ylim(0.22, 0.31)
    plt.legend(loc="upper right", fontsize=legend_size)
    plt.savefig("figure/limit参数敏感性.png")
    fig.savefig("figure/limit参数敏感性.eps", dpi=600, format='eps')
    plt.show()


def negative_sample():
    limit = np.arange(0.1, 1, 0.1)
    neg = np.arange(5, 35, 5)

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in neg:
        temp = []
        for p1 in limit:
            filename = "2007_2016_embs_limit_"+str(round(p1, 1))+"_neg_"+str(int(p0))+"_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T

    index = 0
    fig = plt.figure()
    for i in range(4):
        plt.plot([int(n) for n in neg], score[i*2+1], label="$m$="+str(limit[i*2+1]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$n$", fontdict={"size": label_size})  # label_size tick_size legend_size
    plt.xticks(fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    # plt.ylim(0.23, 0.31)
    plt.legend(loc="upper right", fontsize=legend_size)
    plt.savefig("figure/neg参数敏感性.png")
    fig.savefig("figure/neg参数敏感性.eps", dpi=600, format='eps')
    plt.show()


def num_walk():
    nWalk = np.arange(10, 110, 10)
    lWalk = np.arange(10, 110, 10)

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in nWalk:
        temp = []
        for p1 in lWalk:
            filename = "2007_2016_embs_nWalk_" + str(int(p0)) + "_lWalk_" + str(int(p1)) + "_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T

    index = 0
    fig = plt.figure()
    for i in range(4):
        plt.plot([int(n) for n in nWalk], score[i*2+1], label="$l$="+str(lWalk[i*2+1]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$r$", fontdict={"size": label_size})  # label_size tick_size legend_size
    plt.xticks(nWalk, fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    # plt.ylim(0.23, 0.31)
    plt.legend(loc="upper right", fontsize=legend_size)
    plt.savefig("figure/nWalk参数敏感性.png")
    fig.savefig("figure/nWalk参数敏感性.eps", dpi=600, format='eps')
    plt.show()

def length_walk():
    nWalk = np.arange(10, 110, 10)
    lWalk = np.arange(10, 110, 10)

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in lWalk:
        temp = []
        for p1 in nWalk:
            filename = "2007_2016_embs_nWalk_" + str(int(p1)) + "_lWalk_" + str(int(p0)) + "_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T

    index = 0
    fig = plt.figure()
    for i in range(4):
        plt.plot([int(n) for n in lWalk], score[i*2+1], label="$r$="+str(lWalk[i*2+1]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$l$", fontdict={"size": label_size})  # label_size tick_size legend_size
    plt.xticks(lWalk, fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    # plt.ylim(0.19, 0.29)
    plt.legend(fontsize=legend_size)
    plt.savefig("figure/lWalk参数敏感性.png")
    fig.savefig("figure/lWalk参数敏感性.eps", dpi=600, format='eps')
    plt.show()


def window_size():
    window = np.arange(2, 22, 2)
    dim = [16, 32, 64, 128, 256, 512]

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in window:
        temp = []
        for p1 in dim:
            filename = "2007_2016_embs_window_" + str(int(p0)) + "_dim_" + str(int(p1)) + "_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T

    index = 0
    fig = plt.figure()
    plt.plot([int(n) for n in window], score[0], label="$d$=" + str(dim[0]), color=colors[index], marker=markers[index])
    index += 1
    for i in range(2, 5):
        plt.plot([int(n) for n in window], score[i], label="$d$="+str(dim[i]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$w$", fontdict={"size": label_size})  # label_size tick_size legend_size
    plt.xticks(window, fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    plt.ylim(0.15, 0.4)
    plt.legend(fontsize=legend_size)
    plt.savefig("figure/window参数敏感性.png")
    fig.savefig("figure/window参数敏感性.eps", dpi=600, format='eps')
    plt.show()


def dim_para():
    window = np.arange(2, 22, 2)
    dim = [16, 32, 64, 128, 256, 512]

    filepath = "parameter_sensitivity/collaborate_network(2G)/recommendation"
    score = []
    for p0 in dim:
        temp = []
        for p1 in window:
            filename = "2007_2016_embs_window_" + str(int(p1)) + "_dim_" + str(int(p0)) + "_G_2ori.txt"
            with open(os.path.join(filepath, filename), "r") as lines:
                for line in lines:
                    line_json = json.loads(line)
                    # temp.append(line_json["top_5"])
                    temp.append(np.mean(list(line_json.values())))
        score.append(temp)
    score = np.array(score).T

    index = 0
    fig = plt.figure()
    for i in range(4):
        plt.plot(np.arange(len(dim)), score[2*i+1], label="$w$="+str(window[2*i+1]), color=colors[index], marker=markers[index])
        index += 1
    plt.xlabel("$d$", fontdict={"size": label_size})  # label_size tick_size legend_size
    plt.xticks(np.arange(len(dim)), [str(d) for d in dim], fontsize=tick_size)
    plt.ylabel("average precision", fontdict={"size": label_size})
    plt.yticks(fontsize=tick_size)
    plt.ylim(0.15, 0.4)
    plt.legend(fontsize=legend_size, loc="upper left")
    plt.savefig("figure/dim参数敏感性.png")
    fig.savefig("figure/dim参数敏感性.eps", dpi=600, format='eps')
    plt.show()


if __name__ == '__main__':
    label_size = 17
    tick_size = 15
    legend_size = 13.5
    colors = ["red", "gold", "green", "cyan", "blue", "purple"]
    markers = ["o", "v", "^", "<", ">", "x"]

    # evaluation()
    # limit_para()
    # negative_sample()
    # num_walk()
    # length_walk()
    # window_size()
    dim_para()
