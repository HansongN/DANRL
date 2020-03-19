# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 15:46 
import json
import string

def Hindex(citel):
    citel.sort(reverse=True)
    if citel[0] == 0:
        return 0
    for i in range(len(citel)):
        if citel[i] < i + 1:
            return i
    return len(citel)


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


def compute_hIndex_citation():
    j = 0
    for sid, name in seed_sid_name.items():
        j += 1
        # print("—读入2016及以前的title及pub_year,并按年份排序")
        pid_year = dict()
        filepath = 'data/seed scholars papers/' + sid + '.txt'
        with open(filepath, 'r') as papers:
            for paper in papers:
                paper_json = json.loads(paper)
                if 'id' in paper_json and 'title' in paper_json and 'year' in paper_json and paper_json["year"] <= 2016:
                    pid_year[paper_json['id']] = paper_json['year']
        papers.close()

        pid_year = dict(sorted(pid_year.items(), key=lambda d:d[1], reverse=False))

        first_pub_year = list(pid_year.values())[0]

        # print("—读入2016以前的{pid: title}")
        pid_title = dict()
        filepath = 'data/seed scholars papers/' + sid + '.txt'
        with open(filepath, 'r') as papers:
            for paper in papers:
                paper_json = json.loads(paper)
                if 'id' in paper_json and 'title' in paper_json and 'year' in paper_json and paper_json["year"] <= 2016:
                    pid_title[paper_json['id']] = paper_json['title'].strip(string.punctuation).lower()
        papers.close()

        # print("—初始化每篇论文的被引次数 {year: [{id, cited_times}]}")
        year_cited_times = dict()  # 被引用的年份
        for year in range(first_pub_year, 2017):
            year_cited_times[year] = list()
            for pid, pub_year in pid_year.items():
                if pub_year <= year:
                    year_cited_times[year].append({"id": pid, "cited_times": 0})

        # print("—按年份计算每篇论文每一年的被引次数 {year: [{id: cited_times}]}")
        # i = 0
        for year, cited_papers in year_cited_times.items():
            # i += 1
            for title, year_references in title_year_ref.items():
                if year_references[0] == year:
                    for cited_paper in cited_papers:
                        if pid_title[cited_paper['id']] in year_references[1]:
                            cited_paper['cited_times'] += 1
            # print("——" + str(i) + '/' + str(len(year_cited_times.values())))

        # print("—按年份计算每篇论文的被引次数 {year: [{title, cited_times}]}")
        last_year_cited_times = list()
        # i = 0
        for year, cited_papers in year_cited_times.items():
            # i += 1
            if year == first_pub_year:
                last_year_cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
            else:
                this_year_cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
                for i in range(len(last_year_cited_times)):
                    this_year_cited_times[i] = this_year_cited_times[i] + last_year_cited_times[i]
                last_year_cited_times = this_year_cited_times
                index = 0
                for cited_paper in cited_papers:
                    cited_paper['cited_times'] = this_year_cited_times[index]
                    index += 1
            # print("——" + str(i) + '/' + str(len(year_cited_times.values())))

        # print("—按年份计算作者总被引次数{year: n_cited} 及 h_index")
        year_nPubs = dict()
        year_nCited = dict()
        year_hIndex = dict()
        # i = 0
        for year, cited_papers in year_cited_times.items():
            # i += 1
            cited_times = [cited_paper['cited_times'] for cited_paper in cited_papers]
            cited_times.sort()
            h_index = Hindex(cited_times)
            n_cited = sum(cited_times)
            n_pubs = len(cited_papers)
            year_nPubs[year] = n_pubs
            year_nCited[year] = n_cited
            year_hIndex[year] = h_index
            # print("——" + str(i) + '/' + str(len(year_cited_times.values())))

        # print("—按年份计算学者每篇论文平均的被引次数")
        year_ave_cited = dict()
        for year, n_pub in year_nPubs.items():
            year_ave_cited[year] = year_nCited[year] / n_pub

        # 写入文件
        # print("—写入文件")
        author_info = dict()
        author_info["id"] = sid
        author_info["name"] = name
        author_info["first_pub_year"] = first_pub_year
        author_info["n_pubs"] = year_nPubs
        author_info["n_cited"] = year_nCited
        author_info["h_index"] = year_hIndex
        author_info["ave_cited_times"] = year_ave_cited
        author_info["paper_cited_times"] = year_cited_times
        output = open('data/seed_scholar_nPub_nCited_hIndex.txt', 'a')
        output.write(json.dumps(author_info) + '\n')
        output.close()
        print(str(j) + '/' + str(len(seed_sid_name.values())))


def read_seed_scholar_id_name():
    id_name = dict()
    with open(r"data/seed_scholars_info.txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            id_name[scholar_json["id"]] = scholar_json["name"]
    return id_name


if __name__ == '__main__':
    print("# 读入seed_scholar的{id: name}")
    seed_sid_name = read_seed_scholar_id_name()

    print("# 读入引用信息中的{title:[year,references]}")
    title_year_ref = read_title_year_references()

    print("# 计算seed scholar的h-index，被引次数等")
    compute_hIndex_citation()

