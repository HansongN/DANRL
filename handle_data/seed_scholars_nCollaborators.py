# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 20:51 
import json

def read_seed_scholar_id_name():
    id_name = dict()
    with open(r"data/seed_scholars_info.txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            id_name[scholar_json["id"]] = scholar_json["name"]
    return id_name


def find_smaller_year(target, year_list):
    y = target
    year_list.sort()
    for year in year_list:
        if year < target:
            y = year
    return y


print("# 读入seed_scholar的{id: name}")
seed_scholar_id_name = read_seed_scholar_id_name()

print("# 按年份读入每个seed_scholar的每个合作者的id")
seed_scholar_year_collaborator_id = dict()
with open("data/seed_scholars_collaborators_id_times_year.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        seed_scholar_year_collaborator_id[scholar_json["id"]] = dict()
        for collaborator in scholar_json["collaborators"]:
            for year in collaborator["collaboration_years"]:
                if year in seed_scholar_year_collaborator_id[scholar_json["id"]]:
                    seed_scholar_year_collaborator_id[scholar_json["id"]][year].append(collaborator["id"])
                else:
                    seed_scholar_year_collaborator_id[scholar_json["id"]][year] = [collaborator["id"]]
scholars.close()

for s_id, year_c_ids in seed_scholar_year_collaborator_id.items():
    seed_scholar_year_collaborator_id[s_id] = dict(sorted(year_c_ids.items(), key=lambda keys: keys[0]))

for s_id, year_c_ids in seed_scholar_year_collaborator_id.items():
    past_collaborator = []
    for year, c_ids in year_c_ids.items():
        if len(past_collaborator) == 0:
            past_collaborator = c_ids
        else:
            values = past_collaborator + c_ids
            year_c_ids[year] = values
            past_collaborator = values

    pub_year = list(year_c_ids.keys())
    first_pub_year = pub_year[0]
    for year in range(first_pub_year + 1, 2017):
        if year not in pub_year:
            t_year = find_smaller_year(year, pub_year)
            year_c_ids[year] = year_c_ids[t_year]
    seed_scholar_year_collaborator_id[s_id] = dict(sorted(year_c_ids.items(), key=lambda keys: keys[0]))


print("# 按年份记录合作者的数量")
output = open("data/seed_scholars_n_Collaborators.txt", "w")
for s_id, year_c_ids in seed_scholar_year_collaborator_id.items():
    temp = dict()
    temp["id"] = s_id
    temp["name"] = seed_scholar_id_name[s_id]
    temp["n_Collaborators"] = dict()
    for year, c_ids in year_c_ids.items():
        temp["n_Collaborators"][year] = len(list(set(c_ids)))
    temp["collaborators"] = dict()
    for year, c_ids in year_c_ids.items():
        temp["collaborators"][year] = list(set(c_ids))
    output.write(json.dumps(temp) + "\n")
output.close()
