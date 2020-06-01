# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 14:43
import os
import json

# aminer_papers_files_path = r"I:\open-academic-graph-2019-01\aminer_papers"
sigmod_paper_path = "data/sigmod_papers_07_16.txt"

# print("提取2007-2016年SIGMOD上的论文信息")
# out_file = open(sigmod_paper_path, 'w')
# files = os.listdir(aminer_papers_files_path)
# for file in files:
#     print(file)
#     filePath = os.path.join(aminer_papers_files_path, file)
#     papers = open(filePath, 'r')
#     for paper in papers:
#         paper_json = json.loads(paper)
#         if 'venue' in paper_json and 'id' in paper_json['venue'] and paper_json['venue'][
#             'id'] == "547ffa8cdabfaebedf84f21b" and "year" in paper_json and paper_json["year"]>=2007 and paper_json["year"] <= 2016:
#             out_file.write(paper)
# out_file.close()

print("计算每个学者07-16年间在SIGMOD上发表论文数量")
author_papers_number = {}
with open(sigmod_paper_path, "r") as papers:
    for paper in papers:
        paper_json = json.loads(paper)
        if "authors" in paper_json:
            for author in paper_json['authors']:
                if 'id' in author and author['id'] in author_papers_number:
                    author_papers_number[author['id']] += 1
                elif 'id' in author:
                    author_papers_number[author['id']] = 1
                else:
                    pass
papers.close()
author_papers_number = dict(sorted(author_papers_number.items(), key=lambda d:d[1], reverse=True))
author_n_paper_path = r"data/author_n_paper_07_16.txt"
out_file = open(author_n_paper_path, 'w')
for key,value in author_papers_number.items():
    temp = dict()
    temp['id'] = key
    temp['n_pubs'] = value
    temp_json = json.dumps(temp)
    out_file.writelines(temp_json+ '\n')
out_file.close()

