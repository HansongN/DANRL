# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/12 20:08 
import json

def find_smaller_year(target, year_list):
    y = target
    year_list.sort()
    for year in year_list:
        if year < target:
            y = year
    return y

print("读入id，姓名")
sid_name = dict()
with open("data/100_seed_scholars_info.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid_name[scholar_json["id"]] = scholar_json["name"]
scholars.close()
with open("data/100_seed_scholars_collaborators_info.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] not in sid_name:
            sid_name[scholar_json["id"]] = scholar_json["name"]
scholars.close()

print("读入id，学术元年，论文数量，被引次数，h-index，平均被引次数")
sid_fpy = dict()
sid_nPub = dict()
sid_nCited = dict()
sid_hIndex = dict()
sid_aveCited = dict()
with open("data/100_seed_scholar_nPub_nCited_hIndex.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid_fpy[scholar_json["id"]] = scholar_json["first_pub_year"]
        sid_nPub[scholar_json["id"]] = scholar_json["n_pubs"]
        sid_nCited[scholar_json["id"]] = scholar_json["n_cited"]
        sid_hIndex[scholar_json["id"]] = scholar_json["h_index"]
        sid_aveCited[scholar_json["id"]] = scholar_json["ave_cited_times"]
scholars.close()
with open("data/100_collaborators_nPub_nCited_hIndex.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] not in sid_fpy:
            sid_fpy[scholar_json["id"]] = scholar_json["first_pub_year"]
            sid_nPub[scholar_json["id"]] = scholar_json["n_pubs"]
            sid_nCited[scholar_json["id"]] = scholar_json["n_cited"]
            sid_hIndex[scholar_json["id"]] = scholar_json["h_index"]
            sid_aveCited[scholar_json["id"]] = scholar_json["ave_cited_times"]
scholars.close()

print("读入id，合作者数量，合作者id")
sid_nCollaborator = dict()
sid_collaborators = dict()
with open("data/100_seed_scholars_n_Collaborators.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid_nCollaborator[scholar_json["id"]] = scholar_json["n_Collaborators"]
        sid_collaborators[scholar_json["id"]] = scholar_json["collaborators"]
scholars.close()
with open("data/100_seed_scholars_collaborators_n_Collaborators.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] not in sid_nCollaborator:
            sid_nCollaborator[scholar_json["id"]] = scholar_json["n_Collaborators"]
            sid_collaborators[scholar_json["id"]] = scholar_json["collaborators"]
scholars.close()

print("读入id，研究领域")
sid_fos = dict()
with open("data/100_seed_scholars_fos.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid_fos[scholar_json["id"]] = scholar_json["fos"]
scholars.close()
with open("data/100_seed_scholars_collaborators_fos.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        if scholar_json["id"] not in sid_fos:
            sid_fos[scholar_json["id"]] = scholar_json["fos"]
scholars.close()

print("写入文件")
output = open("data/100_scholars_attribute.txt", "w")
for sid, name in sid_name.items():
    if sid in sid_fpy:
        temp = dict()
        temp["id"] = sid
        temp["name"] = name
        temp["first_pub_year"] = sid_fpy[sid]
        temp["n_pub"] = sid_nPub[sid]
        temp["n_cited"] = sid_nCited[sid]
        temp["h_index"] = sid_hIndex[sid]
        temp["ave_cited"] = sid_aveCited[sid]
        temp["n_collaborator"] = sid_nCollaborator[sid]
        temp["collaborators"] = sid_collaborators[sid]
        temp["fos"] = sid_fos[sid]
        output.write(json.dumps(temp) + "\n")
output.close()

print("按年份写入文件")
sid_year = dict()
with open("data/100_scholars_attribute.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        fpy = scholar_json["first_pub_year"]
        years = list(scholar_json["n_pub"].keys())
        sid_year[scholar_json["id"]] = dict()
        temp = dict()
        for year in years:
            temp[year] = dict()
            temp[year]["academic_age"] = int(year) - int(fpy) + 1
            # temp[year] = dict()
            temp[year]["n_pub"] = scholar_json["n_pub"][year]
            temp[year]["n_cited"] = scholar_json["n_cited"][year]
            temp[year]["h_index"] = scholar_json["h_index"][year]
            temp[year]["ave_cited"] = scholar_json["ave_cited"][year]
            if year in scholar_json["n_collaborator"]:
                temp[year]["n_collaborator"] = scholar_json["n_collaborator"][year]
            else:
                temp[year]["n_collaborator"] = 0
            if year in scholar_json["collaborators"]:
                temp[year]["collaborators"] = scholar_json["collaborators"][year]
            else:
                temp[year]["collaborators"] = []
            temp[year]["fos"] = scholar_json["fos"][year]

        years = list(temp.keys())
        years = [int(y) for y in years]
        for y in range(years[0], 2017):
            if y not in years:
                t_year = find_smaller_year(y, years)
                temp[y] = temp[t_year]
        temp = dict(sorted(temp.items(), key=lambda d: d[0], reverse=False))
        sid_year[scholar_json["id"]] = temp
scholars.close()

print("按年份写入文件")
for sid, info in sid_year.items():
    years = list(info.keys())
    for year in years:
        output = open("data/100 scholars attribute by year/" + str(year) + ".txt", "a")
        temp = dict()
        temp["id"] = sid
        temp["name"] = sid_name[sid]
        temp["academic_age"] = info[year]["academic_age"]
        # temp[year] = dict()
        temp["n_pub"] = info[year]["n_pub"]
        temp["n_cited"] = info[year]["n_cited"]
        temp["h_index"] = info[year]["h_index"]
        temp["ave_cited"] = info[year]["ave_cited"]
        temp["n_collaborator"] = info[year]["n_collaborator"]
        temp["collaborators"] = info[year]["collaborators"]
        temp["fos"] = info[year]["fos"]
        output.write(json.dumps(temp) + "\n")
        output.close()

