# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/13 10:18
import json

num_seed_scholar = 100

print("# 读入seed scholars的id")
seed_scholars_id = list()
with open('data/author_n_paper_07_16.txt', 'r') as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if len(seed_scholars_id) < num_seed_scholar:
            seed_scholars_id.append(scholar_json['id'])
scholars.close()

output = open("data/" + str(num_seed_scholar) + "_seed_scholars_info.txt", "w")
with open('data/seed_scholars_info.txt', 'r') as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] in seed_scholars_id:
            output.write(scholar)
scholars.close()
output.close()

output = open("data/" + str(num_seed_scholar) + "_seed_scholar_nPub_nCited_hIndex.txt", "w")
with open('data/seed_scholar_nPub_nCited_hIndex.txt', 'r') as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] in seed_scholars_id:
            output.write(scholar)
scholars.close()
output.close()

output = open("data/" + str(num_seed_scholar) + "_seed_scholars_n_Collaborators.txt", "w")
with open('data/seed_scholars_n_Collaborators.txt', 'r') as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] in seed_scholars_id:
            output.write(scholar)
scholars.close()
output.close()

output = open("data/" + str(num_seed_scholar) + "_seed_scholars_fos.txt", "w")
with open('data/seed_scholars_fos.txt', 'r') as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] in seed_scholars_id:
            output.write(scholar)
scholars.close()
output.close()

# output = open("data/100_seed_scholars_collaborators_id_times_year.txt", "w")
# with open('data/seed_scholars_collaborators_id_times_year.txt', 'r') as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         if scholar_json["id"] in seed_scholars_id:
#             output.write(scholar)
# scholars.close()
# output.close()
#
# print("# 按年份读入每个seed_scholar的每个合作者的id")
# collaborator_id = list()
# with open("data/100_seed_scholars_collaborators_id_times_year.txt", "r") as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         collaborator_id += [t["id"] for t in scholar_json["collaborators"]]
# collaborator_id = list(set(collaborator_id))
# scholars.close()
#
# output = open("data/100_seed_scholars_collaborators_info.txt", "w")
# with open('data/seed_scholars_collaborators_info.txt', 'r') as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         if scholar_json["id"] in collaborator_id:
#             output.write(scholar)
# scholars.close()
# output.close()

