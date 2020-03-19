# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 15:17
import json
import os
import string
import math

# num_seed_scholar = 500

# print("# 读入seed scholars的id")
# seed_scholars_id = list()
# with open('data/author_n_paper_07_16.txt', 'r') as scholars:
#     for scholar in scholars:
#         scholar_json = json.loads(scholar)
#         if len(seed_scholars_id) < num_seed_scholar:
#             seed_scholars_id.append(scholar_json['id'])
# scholars.close()

# print("# 读入AMiner学者的信息并写入文件")
# output = open("data/seed_scholars_info.txt", "w")
# aminer_authors_filepath = r"I:\open-academic-graph-2019-01\aminer_authors"
# files = os.listdir(aminer_authors_filepath)
# for file in files:
#     filepath = os.path.join(aminer_authors_filepath, file)
#     with open(filepath, 'r') as authors:
#         for author in authors:
#             author_json = json.loads(author)
#             if "id" in author_json and author_json["id"] in seed_scholars_id:
#                 output.write(author)
#     authors.close()
#     print(file + " /20")
# output.close()

print("# 读入学者id")
sid = list()
with open(r"data/seed_scholars_info.txt", "r") as scholars:
    for scholar in scholars:
        scholar_json = json.loads(scholar)
        sid.append(scholar_json["id"])
scholars.close()

print("# 读入有引用信息的论文title")
citation_id = []
with open(r'I:\dblp-ref\aminer_citations_title.txt', 'r') as citations:
    for citation in citations:
        citation_json = json.loads(citation)  # 题目中已去除标点符号，已转小写
        citation_id.append(citation_json['title'])
citations.close()

print("# 提取论文信息")
papers_id = list()
output = open("data/seed_scholars_papers_info.txt", "w")
aminer_papers_filepath = r"I:\open-academic-graph-2019-01\aminer_papers"
files = os.listdir(aminer_papers_filepath)
for file in files:
    filepath = os.path.join(aminer_papers_filepath, file)
    with open(filepath, 'r') as papers:
        for paper in papers:
            paper_json = json.loads(paper)
            if "authors" in paper_json:
                for author in paper_json["authors"]:
                    if "id" in author and author["id"] in sid and "title" in paper_json and paper_json['title'].strip(
                    string.punctuation).lower() in citation_id and "id" in paper_json and paper_json["id"] not in papers_id and "year" in paper_json and paper_json["year"]<=2018:
                        papers_id.append(paper_json["id"])
                        output.write(paper)
                        break
                
    papers.close()
    print(file + "/15")
output.close()

# 列表切片
sid_list = list()
size = math.ceil(len(sid) / 8)
for i in range(8):
    start = i * size
    end = (i + 1)*size if (i + 1)*size < len(sid) else len(sid)
    temp = sid[start : end]
    sid_list.append(temp)

# 找到每个学者个人的论文信息，并写入文件
# s = 1
print("找到每个学者个人的论文信息")
for s in range(8):
    i = 0
    for author_id in sid_list[s]:
        i += 1
        filepath = r'data/seed scholars papers/' + author_id + '.txt'
        output = open(filepath, 'w')
        with open("data/seed_scholars_papers_info.txt", "r") as papers:
            for paper in papers:
                paper_json = json.loads(paper)
                for author in paper_json["authors"]:
                    if "id" in author and author["id"] == author_id:
                        output.write(paper)
                        break
        papers.close()
        output.close()
        print("编号" + str(s) + "：" + str(i) + "/" + str(len(sid_list[s])))