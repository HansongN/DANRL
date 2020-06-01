# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 16:34
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
    papers_filepath = "data/seed scholars papers/" + scholar_id + ".txt"
    collaborator_year = dict()
    # 读入2016年及以前建立的合作关系及合作次数、合作时间
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


"""
    main()
"""
# print("# 读入seed_scholar的{id: name}")
# seed_scholar_id_name = read_seed_scholar_id_name()

# print("# 计算每个seed_scholar与每个合作者的合作次数、合作时间，写入文件")
# output = open("data/seed_scholars_collaborators_id_times_year.txt", "w")
# i = 0
# for scholar_id, scholar_name in seed_scholar_id_name.items():
#     i += 1
#     cid_times_years = get_collaborator_id_times_year(scholar_id)
#     temp = dict()
#     temp["id"] = scholar_id
#     temp["name"] = scholar_name
#     temp["collaborators"] = list()
#     for cid, times_years in cid_times_years.items():
#         temp["collaborators"].append({"id": cid, "collaboration_times": times_years["collaboration_times"], "collaboration_years": times_years["collaboration_years"]})
#     output.write(json.dumps(temp) + "\n")
#     print(str(i) + "/" + str(len(seed_scholar_id_name)))
# output.close()

print("# 按年份读入每个seed_scholar的每个合作者的id")
collaborator_id = list()
seed_scholar_year_collaborator_id = dict()
sid_cid = dict()
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
        collaborator_id += [t["id"] for t in scholar_json["collaborators"]]
        sid_cid[scholar_json["id"]] = [t["id"] for t in scholar_json["collaborators"]]
collaborator_id = list(set(collaborator_id))
scholars.close()

# print("# 按年份记录合作者的数量")
# output = open("data/seed_scholars_n_Collaborators.txt", "w")
# for s_id, year_c_ids in seed_scholar_year_collaborator_id.items():
#     temp = dict()
#     temp["id"] = s_id
#     temp["name"] = seed_scholar_id_name[s_id]
#     temp["n_Collaborators"] = dict()
#     for year, c_ids in year_c_ids.items():
#         temp["n_Collaborators"][year] = len(list(set(c_ids)))
#     output.write(json.dumps(temp) + "\n")

# print("# 读入已记录的合作者id")
# collaborator_id_read = list()
# with open("data/seed_scholars_collaborators_info.txt", "r") as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         collaborator_id_read.append(scholar_json["id"])
# scholars.close()
# collaborator_id_read = list(set(collaborator_id_read))

# print("读入每个合作者的信息")
# index = 14
# output = open("data/seed_scholars_collaborators_info_" + str(index) + ".txt", "a")
# aminer_authors_filepath = r"I:\open-academic-graph-2019-01\aminer_authors"
# files = os.listdir(aminer_authors_filepath)
# for file in files:

# file_path = os.path.join(aminer_authors_filepath, files[index])
# with open(file_path, 'r') as authors:
#     for author in authors:
#         author_json = json.loads(author)
#         if "id" in author_json and author_json["id"] in collaborator_id and author_json["id"] not in collaborator_id_read:
#             output.write(author)
# authors.close()
# print(files[14] + "/20")
# output.close()
#
# output = open("data/seed_scholars_collaborators_info.txt", "a")
# for i in range(14, 20):
#     with open("data/seed_scholars_collaborators_info_" + str(index) + ".txt", "r") as authors:
#         for author in authors:
#             output.write(author)
#     authors.close()
# output.close()

# print("找到每个学者的合作者信息，并写入文件")
# i = 0
# for s_id, c_ids in sid_cid.items():
#     i += 1
#     filepath = 'data/seed scholars collaborators/' + s_id + '.txt'
#     output = open(filepath, 'w')
#     with open("data/seed_scholars_collaborators_info.txt", "r") as scholars:
#         for scholar in scholars:
#             scholar_json = json.loads(scholar)
#             if scholar_json["id"] in c_ids:
#                 output.write(scholar)
#     output.close()
#     print("已完成：" + str(i) + "/" + str(len(sid_cid.values())))

print("按年份记录合作者的id")
year_sid_cid = {}
for s_id, year_c_ids in seed_scholar_year_collaborator_id.items():
    for year, c_ids in year_c_ids.items():
        if year not in year_sid_cid.keys():
            year_sid_cid[year] = [[s_id, c_ids[0]]]
            for i in range(1, len(c_ids)):
                if [s_id, c_ids[i]] not in year_sid_cid[year]:
                    year_sid_cid[year].append([s_id, c_ids[i]])
        else:
            for i in range(len(c_ids)):
                if [s_id, c_ids[i]] not in year_sid_cid[year]:
                    year_sid_cid[year].append([s_id, c_ids[i]])

print("找到每个学者每年的合作者信息，并写入文件")
for year, sid_cid in year_sid_cid.items():
    filepath = 'data/collaboration_record/' + str(year) + '.txt'
    output = open(filepath, 'w')
    for sc in sid_cid:
        output.write(sc[0] + " " + sc[1] + "\n")
    output.close()
#
# # 读入有引用信息的title-references
# print("读入引用信息title-references")
# citation_id = dict()
# with open(r'I:\dblp-ref\aminer_citations_title.txt', 'r') as citations:
#     for citation in citations:
#         citation_json = json.loads(citation)  # 题目中已去除标点符号，已转小写
#         citation_id[citation_json['title']] = citation_json['references']
# citations.close()
#
# # 提取论文信息
# print("提取AMINER论文信息")
# papers_id_list = list()
# output = open("data/collaborators_papers_info.txt", "w")
# aminer_papers_filepath = r"I:\open-academic-graph-2019-01\aminer_papers"
# files = os.listdir(aminer_papers_filepath)
# for file in files:
#     file_path = os.path.join(aminer_papers_filepath, file)
#     with open(file_path, 'r') as papers:
#         for paper in papers:
#             paper_json = json.loads(paper)
#             if "authors" in paper_json:
#                 for author in paper_json["authors"]:
#                     if "id" in author and author["id"] in collaborator_id and "title" in paper_json and paper_json['title'].strip(
#                             string.punctuation).lower() in citation_id and "id" in paper_json and paper_json[
#                         "id"] not in papers_id_list and "year" in paper_json and paper_json["year"] <= 2018:
#                         papers_id_list.append(paper_json["id"])
#                         output.write(paper)
#                         break
#     papers.close()
#     print(file + "/15")
# output.close()
# #
# print("# 读入合作者的id")
# collaborator_id = list()
# with open("data/seed_scholars_collaborators_info.txt", "r") as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         collaborator_id.append(scholar_json["id"])
# print(len(set(collaborator_id)))
# scholars.close()
#
# # 列表切片
# collaborator_id_list = list()
# size = math.ceil(len(collaborator_id) / 8)
# for i in range(8):
#     start = i * size
#     end = (i + 1) * size if (i + 1) * size < len(collaborator_id) else len(collaborator_id)
#     temp = collaborator_id[start: end]
#     collaborator_id_list.append(temp)
#
# temp = set()
# r = "G:\AMiner_data\seed scholars collaborators papers"
# f = os.listdir(r)
# for f0 in f:
#     temp.add(f0[:-4])
#
# # 找到每个学者个人的论文信息，并写入文件
# # s = 7
# print("找到每个学者个人的论文信息")
# for s in range(8):
#     i = 0
#     for author_id in collaborator_id_list[s]:
#         i += 1
#         if author_id not in temp:
#             filepath = 'data/seed scholars collaborators papers/' + author_id + '.txt'
#             output = open(filepath, 'w')
#             with open("data/collaborators_papers_info.txt", "r") as papers:
#                 for paper in papers:
#                     paper_json = json.loads(paper)
#                     for author in paper_json["authors"]:
#                         if "id" in author and author["id"] == author_id:
#                             output.write(paper)
#                             break
#             papers.close()
#             output.close()
#         else:
#             filepath = 'data/seed scholars collaborators papers/' + author_id + '.txt'
#             read_pid = list()
#             with open(filepath, "r") as papers:
#                 for paper in papers:
#                     paper_json = json.loads(paper)
#                     read_pid.append(paper_json["id"])
#             papers.close()
#
#             output = open(filepath, 'a')
#             with open(r"data/collaborators_papers_info.txt", "r") as papers:
#                 for paper in papers:
#                     paper_json = json.loads(paper)
#                     for author in paper_json["authors"]:
#                         if "id" in author and author["id"] == author_id and paper_json["id"] not in read_pid:
#                             output.write(paper)
#                             break
#             papers.close()
#             output.close()
#         print("编号" + str(s) + "：" + str(i) + "/" + str(len(collaborator_id_list[s])))