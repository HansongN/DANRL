# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 20:09

import math
import json
import string
from multiprocessing import Pool


def Hindex(citel):
    citel.sort(reverse=True)
    if citel[0] == 0:
        return 0
    for i in range(len(citel)):
        if citel[i] < i + 1:
            return i
    return len(citel)


def compute_hIndex_citation(cid_name_list, findex, filepathL, title_year_ref0):
    c_cid_list = list()
    # with open(filepathL[findex], 'r') as scholars:
    #     for scholar in scholars:
    #         scholar_json = json.loads(scholar)
    #         c_cid_list.append(scholar_json["id"])
    # scholars.close()

    collaborator_id_name = cid_name_list[findex]
    output = open(filepathL[findex], 'a')
    j = 0
    for cid, name in collaborator_id_name.items():
        j += 1
        if cid not in com_sid and cid not in c_cid_list:
            # print("—读入2016及以前的title及pub_year,并按年份排序")
            pid_year = dict()
            filepath = 'data/seed scholars collaborators papers/' + cid + '.txt'
            with open(filepath, 'r') as papers:
                for paper in papers:
                    paper_json = json.loads(paper)
                    if 'id' in paper_json and 'title' in paper_json and 'year' in paper_json and paper_json["year"] <= 2016:
                        pid_year[paper_json['id']] = paper_json['year']
            papers.close()
            if len(pid_year.keys()) > 0:
                pid_year = dict(sorted(pid_year.items(), key=lambda d:d[1], reverse=False))
                first_pub_year = list(pid_year.values())[0]

                # print("—读入2016及以前的{pid: title}")
                pid_title = dict()
                filepath = 'data/seed scholars collaborators papers/' + cid + '.txt'
                with open(filepath, 'r') as papers:
                    for paper in papers:
                        paper_json = json.loads(paper)
                        if 'id' in paper_json and 'title' in paper_json and 'year' in paper_json and paper_json[
                            "year"] <= 2016:
                            pid_title[paper_json['id']] = paper_json['title'].strip(string.punctuation).lower()
                papers.close()

                # print("—初始化每篇论文的被引次数 {year: [{title, cited_times}]}")
                year_cited_times = dict()  # 被引用的年份
                for year in range(first_pub_year, 2017):
                    year_cited_times[year] = list()
                    for pid, pub_year in pid_year.items():
                        if pub_year <= year:
                            year_cited_times[year].append({"id": pid, "cited_times": 0})

                # print("—按年份计算每篇论文每一年的被引次数 {year: [{title, cited_times}]}")
                for year, cited_papers in year_cited_times.items():
                    for title, year_references in title_year_ref0.items():
                        if year_references[0] == year:
                            for cited_paper in cited_papers:
                                if pid_title[cited_paper['id']] in year_references[1]:
                                    cited_paper['cited_times'] += 1

                # print("—按年份计算每篇论文的被引次数 {year: [{title, cited_times}]}")
                last_year_cited_times = list()
                for year, cited_papers in year_cited_times.items():
                    if year == first_pub_year:
                        last_year_cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
                    else:
                        this_year_cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
                        for k in range(len(last_year_cited_times)):
                            this_year_cited_times[k] = this_year_cited_times[k] + last_year_cited_times[k]
                        last_year_cited_times = this_year_cited_times
                        index = 0
                        for cited_paper in cited_papers:
                            cited_paper['cited_times'] = this_year_cited_times[index]
                            index += 1

                # print("—按年份计算作者总被引次数{year: n_cited} 及 h_index")
                year_nPubs = dict()
                year_nCited = dict()
                year_hIndex = dict()
                for year, cited_papers in year_cited_times.items():
                    cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
                    cited_times.sort()
                    h_index = Hindex(cited_times)
                    n_cited = sum(cited_times)
                    n_pubs = len(cited_papers)
                    year_nPubs[year] = n_pubs
                    year_nCited[year] = n_cited
                    year_hIndex[year] = h_index

                # print("—按年份计算学者每篇论文平均的被引次数")
                year_ave_cited = dict()
                for year, n_pub in year_nPubs.items():
                    year_ave_cited[year] = year_nCited[year] / n_pub

                # print("—写入文件")
                author_info = dict()
                author_info["id"] = cid
                author_info["name"] = name
                author_info["first_pub_year"] = first_pub_year
                author_info["n_pubs"] = year_nPubs
                author_info["n_cited"] = year_nCited
                author_info["h_index"] = year_hIndex
                author_info["ave_cited_times"] = year_ave_cited
                author_info["paper_cited_times"] = year_cited_times
                output.write(json.dumps(author_info) + '\n')
        print("进程" + str(findex) + ": " + str(j) + '/' + str(len(collaborator_id_name.values())))
    output.close()


# 字典切片
def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice


def read_seed_scholar_collaborator_id_name():
    id_name = dict()
    with open("data/seed_scholars_collaborators_info.txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            id_name[scholar_json["id"]] = scholar_json["name"]
    return id_name


def read_title_year_references():
    title_year_ref = dict()
    with open(r'I:\dblp-ref\aminer_citations_title.txt', 'r') as citations:
        for citation in citations:
            citation_json = json.loads(citation)
            if 'year' in citation_json:
                title = citation_json['title'].strip(string.punctuation).lower()  # 去除标点符号，转小写
                title_year_ref[title] = [citation_json['year'], citation_json['references']]
    citations.close()
    return title_year_ref


if __name__ == '__main__':
    # print("# 读入合作者的id_name")
    # collaborator_id_name0 = read_seed_scholar_collaborator_id_name()
    # print(len(collaborator_id_name0.keys()))
    # #
    # print("# 读入引用信息中的{title:[year,references]}")
    # title_year_ref = read_title_year_references()

    # com_sid = []
    # with open("data/100_collaborators_nPub_nCited_hIndex.txt", "r") as scholars:
    #     for scholar in scholars:
    #         scholar_json = json.loads(scholar)
    #         com_sid.append(scholar_json["id"])
    # scholars.close()

    # print("字典切片")
    # collaborator_id_name_list = list()
    # for i in range(8):
    #     size = math.ceil(len(collaborator_id_name0.keys()) / 8)
    #     collaborator_id_name_list.append(dict_slice(collaborator_id_name0, i * size,
    #                                           (i + 1) * size if (i + 1) * size < len(
    #                                               collaborator_id_name0.values()) else len(
    #                                               collaborator_id_name0.values())))

    filepath_list = ["data/collaborators_nPub_nCited_hIndex_0.txt",
                     "data/collaborators_nPub_nCited_hIndex_1.txt",
                     "data/collaborators_nPub_nCited_hIndex_2.txt",
                     "data/collaborators_nPub_nCited_hIndex_3.txt",
                     "data/collaborators_nPub_nCited_hIndex_4.txt",
                     "data/collaborators_nPub_nCited_hIndex_5.txt",
                     "data/collaborators_nPub_nCited_hIndex_6.txt",
                     "data/collaborators_nPub_nCited_hIndex_7.txt"]
    # print("# 计算合作者的h-index，被引次数等")
    # print("多进程")
    # p = Pool(8)
    # for pi in range(8):
    #     p.apply_async(compute_hIndex_citation, args=(collaborator_id_name_list, pi, filepath_list, title_year_ref))
    # p.close()
    # p.join()
    # for i in range(8):
    # index = 7
    # compute_hIndex_citation(collaborator_id_name_list, index, filepath_list, title_year_ref)

    print("# 整合文件")
    cid_set = set()
    filepath_list.append("data/100_collaborators_nPub_nCited_hIndex.txt")
    output = open("data/collaborators_nPub_nCited_hIndex.txt", "w")
    for filepath in filepath_list:
        with open(filepath, "r") as lines:
            for line in lines:
                line_json = json.loads(line)
                if line_json["id"] not in cid_set:
                    cid_set.add(line_json["id"])
                    output.write(line)
        lines.close()
    output.close()
    print(len(cid_set))

