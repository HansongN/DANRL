# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 21:27
import json
import os
import math
import string

def read_seed_scholar_id_name():
    id_name = dict()
    with open(r"data/seed_scholars_info.txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            id_name[scholar_json["id"]] = scholar_json["name"]
    return id_name

def get_collaborator_id_times_year(scholar_id):
    papers_filepath = "data/seed scholars collaborators papers/" + scholar_id + ".txt"
    pid_year = dict()
    with open(papers_filepath, 'r') as papers:
        for paper in papers:
            paper_json = json.loads(paper)
            if 'id' in paper_json and 'title' in paper_json and 'year' in paper_json and paper_json["year"] <= 2016:
                pid_year[paper_json['id']] = paper_json['year']
    papers.close()
    if len(pid_year.keys()) > 0:
        collaborator_year = dict()
        with open(papers_filepath, 'r') as papers:
            for paper in papers:
                line_json = json.loads(paper)
                if "id" in line_json and "year" in line_json and "title" in line_json and 'authors' in line_json:
                    for author in line_json['authors']:
                        if 'id' in author and author['id'] != scholar_id:
                            if author['id'] not in collaborator_year:
                                collaborator_year[author['id']] = [line_json["year"]]
                            else:
                                collaborator_year[author['id']].append(line_json["year"])
        papers.close()
        collaborator_times_year = dict()
        for cid, years in collaborator_year.items():
            collaborator_times_year[cid] = dict()
            collaborator_times_year[cid]["collaboration_times"] = len(years)
            collaborator_times_year[cid]["collaboration_years"] = sorted(years)
        collaborator_times_year = dict(
            sorted(collaborator_times_year.items(), key=lambda d: d[1]["collaboration_times"], reverse=True))
        return collaborator_times_year
    else:
        return None


def read_seed_scholar_collaborator_id_name():
    id_name = dict()
    with open("data/seed_scholars_collaborators_info.txt", "r") as scholars:
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

print("# 读入合作者的{id: name}")
seed_scholar_id_name = read_seed_scholar_collaborator_id_name()

print("# 计算每个seed_scholar与每个合作者的合作次数、合作时间，写入文件")
output = open("data/seed_scholars_collaborators_collaborators_id_times_year.txt", "w")
i = 0
for scholar_id, scholar_name in seed_scholar_id_name.items():
    i += 1
    cid_times_years = get_collaborator_id_times_year(scholar_id)
    if cid_times_years != None:
        temp = dict()
        temp["id"] = scholar_id
        temp["name"] = scholar_name
        temp["collaborators"] = list()
        for cid, times_years in cid_times_years.items():
            temp["collaborators"].append({"id": cid, "collaboration_times": times_years["collaboration_times"], "collaboration_years": times_years["collaboration_years"]})
        output.write(json.dumps(temp) + "\n")
    print(str(i) + "/" + str(len(seed_scholar_id_name)))
output.close()

print("# 按年份读入每个seed_scholar的每个合作者的id")
seed_scholar_year_collaborator_id = dict()
with open("data/seed_scholars_collaborators_collaborators_id_times_year.txt", "r") as scholars:
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
    if len(pub_year) > 0:
        first_pub_year = pub_year[0]
        for year in range(first_pub_year + 1, 2017):
            if year not in pub_year:
                t_year = find_smaller_year(year, pub_year)
                year_c_ids[year] = year_c_ids[t_year]
        seed_scholar_year_collaborator_id[s_id] = dict(sorted(year_c_ids.items(), key=lambda keys: keys[0]))


print("# 按年份记录合作者的数量")
output = open("data/seed_scholars_collaborators_n_Collaborators.txt", "w")
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
