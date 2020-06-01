# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/12 21:52 
import json
import os

sid_cid_collTimes = dict()
with open("data/100_seed_scholars_collaborators_id_times_year.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid_cid_collTimes[scholar_json["id"]] = dict()
        for co in scholar_json["collaborators"]:
            sid_cid_collTimes[scholar_json["id"]][co["id"]] = dict()
            years = co["collaboration_years"]
            for year in years:
                if year not in sid_cid_collTimes[scholar_json["id"]][co["id"]]:
                    sid_cid_collTimes[scholar_json["id"]][co["id"]][year] = sum(i < year for i in years)
scholars.close()


for year in range(2006, 2017):
    print(year)
    sid_attribute = {}
    with open("data/100 scholars attribute by year/" + str(year) + ".txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            sid_attribute[scholar_json["id"]] = [scholar_json["academic_age"],
                                                 scholar_json["n_pub"],
                                                 scholar_json["n_cited"],
                                                 scholar_json["h_index"],
                                                 scholar_json["n_collaborator"]]
    scholars.close()

    output = open("data/collaboration_network/" + str(year) + ".txt", "w")
    with open("data/collaboration_record/" + str(year) + ".txt", "r") as lines:
        for line in lines:
            l = line.split()
            s1, s2 = l[0], l[1]
            # print(year, s1, s2)
            if s1 in sid_attribute and s2 in sid_attribute and sid_attribute[s1][0] >3 and sid_attribute[s2][0] >3 and s1 in sid_cid_collTimes:
                temp = dict()
                temp[s1] = sid_attribute[s1]
                temp[s2] = sid_attribute[s2]
                temp["weight"] = sid_cid_collTimes[s1][s2][year] + 1
                output.write(json.dumps(temp) + "\n")
    lines.close()
    output.close()
