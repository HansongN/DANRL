from utils import load_any_obj_pkl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def load_OpenNE_Embedding(method, year):
    sid_emb = dict()
    with open("stability/" + method + "/" + str(year) + "_embs_dim_2.txt", "r") as embeddings:
        embeddings.readline()
        for embedding in embeddings:
            l = embedding.split()
            sid_emb[l[0]] = [float(n) for n in l[1:]]
    embeddings.close()
    return sid_emb


def dis(d):
    dist = 0
    l = len(d)
    for i in range(l-1):
        for j in range(i+1, l):
            dist += np.linalg.norm(np.array(d[i])-np.array(d[j]))
    print(dist)
    return dist


def danrl_plot():
    i = 0
    fig = plt.figure()
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] - 1, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xlabel("$t$=0", fontdict={"size": 12})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    i = 1
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1] - 1, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]-10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=1", fontdict={"size": 12})

    i = 2
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=2", fontdict={"size": 12})

    i = 3
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 3, danrl[i][nodes[j]][1]-10, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=3", fontdict={"size": 12})

    fig.tight_layout(pad=1, w_pad=1.0, h_pad=0.5)

    plt.savefig("stability/figure/danrl.png")
    fig.savefig("stability/figure/danrl.eps", dpi=600, format='eps')

    plt.show()


def deepwalk_plot():
    danrl = deepwalk
    fig = plt.figure()
    i = 0
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] - 10, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=0", fontdict={"size": 12})

    i = 1
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1]+5, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=1", fontdict={"size": 12})

    i = 2
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]-7, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 3, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] - 10, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=2", fontdict={"size": 12})

    i = 3
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1] + 3, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 3, danrl[i][nodes[j]][1] + 3, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=3", fontdict={"size": 12})

    fig.tight_layout(pad=1, w_pad=1.0, h_pad=0.5)
    plt.savefig("stability/figure/deepwalk.png")
    fig.savefig("stability/figure/deepwalk.eps", dpi=600, format='eps')

    plt.show()

def tadw_plot():
    danrl = tadw
    fig = plt.figure()
    i = 0
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 3, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=0", fontdict={"size": 12})

    i = 1
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 1, danrl[i][nodes[j]][1]-10, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]-3, danrl[i][nodes[j]][1] + 3, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=1", fontdict={"size": 12})

    i = 2
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]-7, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]-8, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1]-8, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0]+2, danrl[i][nodes[j]][1]+2, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("$t$=2", fontdict={"size": 12})

    i = 3
    plt.subplot(2, 2, i + 1)
    # for j in range(len(nodes)):
    j = 0
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 1
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 8, danrl[i][nodes[j]][1] + 3, j, fontdict={"size": 15})
    j = 2
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 2, danrl[i][nodes[j]][1] + 2, j, fontdict={"size": 15})
    j = 3
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] - 10, danrl[i][nodes[j]][1] - 8, j, fontdict={"size": 15})
    j = 4
    plt.scatter(danrl[i][nodes[j]][0], danrl[i][nodes[j]][1], c=colors[j], s=80)
    plt.text(danrl[i][nodes[j]][0] + 3, danrl[i][nodes[j]][1] + 3, j, fontdict={"size": 15})
    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("$t$=3", fontdict={"size": 12})
    fig.tight_layout(pad=1, w_pad=1.0, h_pad=0.5)
    plt.savefig("stability/figure/tadw.png")
    fig.savefig("stability/figure/tadw.eps", dpi=600, format='eps')

    plt.show()


if __name__ == '__main__':
    n_g = 4
    # nodes = ['53f34fa1dabfae4b34943ff2', '53f46f94dabfaedd74e8e01f', '53f479f0dabfae8a6845cbb4',
    #          '53f48077dabfae963d25a578', '53f49220dabfaeb22f571d94', '53f450f2dabfaee02ad43a86']
    nodes = ['53f43a14dabfaeee229cd0f1', '53f43605dabfaedce552be3f', '53f48ddcdabfaea7cd1d4a03',
             '53f48051dabfae963d25973f', '53f4a6abdabfaedd74eb7df0']
    colors = ["red", "gold", "green", "cyan", "blue", "purple", "gray", "gray", "gray", "gray", ]

    # danrl = load_any_obj_pkl("stability/2007_2016_embs_dim_2.pkl")
    danrl = load_any_obj_pkl("stability/danrl_2007_2016_embs_dim_2.pkl")
    deepwalk = load_any_obj_pkl("stability/deepwalk_2007_2016_embs_dim_2.pkl")
    tadw = load_any_obj_pkl("stability/tadw_2007_2016_embs_dim_2.pkl")
    # deepwalk = []
    # tadw = []
    # for year in range(2007, 2017):
    #     deepwalk.append(load_OpenNE_Embedding("DeepWalk", year))
    #     tadw.append(load_OpenNE_Embedding("TADW", year))

    # for index in range(len(danrl)):
    #     for key, value in danrl[index].items():
    #         print(key, value)
    #     print(index)
    #
    # nodes = set()
    # for index in range(len(danrl)):
    #     if index == 0:
    #         nodes = set(danrl[index].keys())
    #     else:
    #         nodes = nodes & set(danrl[index].keys())

    # x_max, y_max = 0, 0
    # distance = dict()
    # for node in nodes:
    #     distance[node] = dis([list(danrl[i][node]) for i in range(n_g)])
    # distance = sorted(distance.items(), key=lambda d:d[1], reverse=False)
    # print(distance)
    # print(x_max, y_max)

    danrl_plot()
    deepwalk_plot()
    tadw_plot()





    # for i in range(n_g):
    #     plt.subplot(2, 2, i+1)
    #     for j in range(len(nodes)):
    #         plt.scatter(deepwalk[i][nodes[j]][0], deepwalk[i][nodes[j]][1], c=colors[j], s=80)
    #         plt.text(deepwalk[i][nodes[j]][0]+0.01, deepwalk[i][nodes[j]][1]-0.03, j, fontdict={"size": 15})
    #         plt.xlim(-60, 60)
    #         plt.ylim(-40, 40)
    #         plt.xticks(fontsize=12)
    #         plt.yticks(fontsize=12)
    # plt.show()
    #
    # for i in range(n_g):
    #     plt.subplot(2, 2, i+1)
    #     for j in range(len(nodes)):
    #         plt.scatter(tadw[i][nodes[j]][0], tadw[i][nodes[j]][1], c=colors[j], s=80)
    #         plt.text(tadw[i][nodes[j]][0]+0.01, tadw[i][nodes[j]][1]-0.03, j, fontdict={"size": 15})
    #         plt.xlim(-60, 60)
    #         plt.ylim(-40, 40)
    #         plt.xticks(fontsize=12)
    #         plt.yticks(fontsize=12)
    # plt.show()

    # for i in range(n_g):
    #     plt.subplot(2, 2, i+1)
    #     for j in range(len(nodes)):
    #         print(tadw[i][nodes[j]][0], tadw[i][nodes[j]][1])
    #         plt.scatter(tadw[i][nodes[j]][0], tadw[i][nodes[j]][1], c=colors[j], s=80)
    #         plt.text(tadw[i][nodes[j]][0]+0.01, tadw[i][nodes[j]][1]-0.03, j, fontdict={"size": 15})
    #         # plt.xlim(-0.5, 3)
    #         # plt.ylim(-0.5, 3)
    #         plt.xticks(fontsize=12)
    #         plt.yticks(fontsize=12)
    # plt.show()
