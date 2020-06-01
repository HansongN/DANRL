# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/14 15:16 
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
from utils import save_any_obj_pkl, load_any_obj_pkl
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def construct_original_graph():
    files = os.listdir("handle_data/data/collaboration_network")

    graphs = list()
    for file in files:
        path = os.path.join("handle_data/data/collaboration_network", file)
        g = nx.Graph()
        l = []
        node_attr = dict()
        with open(path, "r") as lines:
            for line in lines:
                line_json = json.loads(line)
                keys = list(line_json.keys())
                s1, s2, w = keys[0], keys[1], line_json[keys[2]]
                a1, a2 = line_json[keys[0]], line_json[keys[1]]
                l.append((s1, s2, w))
                if s1 not in node_attr:
                    node_attr[s1] = a1
                if s2 not in node_attr:
                    node_attr[s2] = a2
        g.add_weighted_edges_from(l)
        attr_list = []
        for node, attr in node_attr.items():
            attr_list.append(attr)
        attr_array = np.array(attr_list)
        attr_normed = attr_array / attr_array.max(axis=0)
        index = 0
        for node, attr in node_attr.items():
            node_attr[node] = list(attr_normed[index])
            index += 1
        for node, attr in node_attr.items():
            g.nodes[node]["attribute"] = attr
        print(g.nodes(data=True))
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        # connected_component_subgraphs(G)
        g = max(nx.connected_component_subgraphs(g), key=len)
        print(file[:-4] + ": #nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        save_any_obj_pkl(g, "graph_data/collaborate_network_" + file[:-4] + ".pkl")

        filename = "graph_data/collaborate_network_" + file[:-4] + "_edgelist.txt"
        nx.write_edgelist(g, filename, data=False)
        nx.draw(g, node_size=20)
        plt.show()
        graphs.append(g)

    save_any_obj_pkl(graphs, "graph_data/collaborate_network_2006_2016.pkl")


def construct_combined_graph():
    graphs = load_any_obj_pkl("graph_data/collaborate_network(1G)/collaborate_network_2006_2016.pkl")
    for i in range(2, len(graphs)):
        g0 = graphs[i-2]
        g1 = graphs[i-1]
        g = graphs[i]
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        l = []
        for edge in g0.edges():
            if edge not in g.edges():
                n1, n2 = edge[0], edge[1]
                l.append((n1, n2, g0.get_edge_data(n1, n2)['weight']))
                if n1 not in g.nodes():
                    g.add_node(n1, attribute=g0.nodes[n1]["attribute"])
                if n2 not in g.nodes():
                    g.add_node(n2, attribute=g0.nodes[n2]["attribute"])

        for edge in g1.edges():
            if edge not in g.edges():
                n1, n2 = edge[0], edge[1]
                l.append((n1, n2, g1.get_edge_data(n1, n2)['weight']))
                if n1 not in g.nodes():
                    g.add_node(n1, attribute=g1.nodes[n1]["attribute"])
                if n2 not in g.nodes():
                    g.add_node(n2, attribute=g1.nodes[n2]["attribute"])
        g.add_weighted_edges_from(l)
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        # nx.draw(g, node_size=20)
        # plt.show()
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        filename = "graph_data/collaborate_network_" + str(i) + "_edgelist_new.txt"
        nx.write_edgelist(g, filename, data=False)

        save_any_obj_pkl(g, "graph_data/collaborate_network(3G)" + str(i+2006) + "_new.pkl")

        graphs.append(g)

    save_any_obj_pkl(graphs, "graph_data/collaborate_network_2008_2016_new.pkl")


def draw_graph():
    graphs = load_any_obj_pkl("graph_data/collaborate_network(2G)/collaborate_network_2007_2016.pkl")
    year = 2007
    G_s = []
    for g in graphs:
        print("#nodes: " + str(g.number_of_nodes()) + ", #edges: " + str(g.number_of_edges()))
        G = nx.Graph()
        G.add_nodes_from(g.nodes())
        G.add_edges_from(g.edges())
        # nx.write_gexf(G, "graph_data/collaborate_network(2G)/" + str(year) + ".gexf")
        # year += 1
        G_s.append(G)
    nx.draw(G_s[0], node_size=30, node_color="black", edge_color="gray")
    plt.show()


if __name__ == '__main__':
    construct_original_graph()
    construct_combined_graph()
    draw_graph()
