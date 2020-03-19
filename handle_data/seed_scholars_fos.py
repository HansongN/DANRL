# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Hansong Nie
# @Time : 2019/12/11 21:53 
import json
import os
import string

def find_smaller_year(target, year_list):
    y = target
    year_list.sort()
    for year in year_list:
        if year < target:
            y = year
    return y


def read_title_fos():
    title_fos = dict()
    with open(r"I:\dblp-ref\aminer_title_fos.txt", 'r') as papers:
        for paper in papers:
            paper_json = json.loads(paper)
            title_fos[paper_json['title']] = paper_json['fos']
    papers.close()
    return title_fos


def scholar_fos(sid):
    rootpath = "data/seed scholars papers"
    filename = sid + ".txt"
    filepath = os.path.join(rootpath, filename)
    title_year_fos = dict()
    # print("读入论文的fos信息")
    with open(filepath, "r") as papers:
        for paper in papers:
            paper_json = json.loads(paper)
            temp = dict()
            temp["year"] = paper_json["year"]
            temp["fos"] = title_fos[paper_json["title"].strip(string.punctuation).lower()]
            title = paper_json["title"].strip(string.punctuation).lower()
            while title in title_year_fos:
                 i = 0
                 title = title + str(i)
            title_year_fos[title] = temp
    # print("排序")
    title_year_fos = dict(sorted(title_year_fos.items(), key=lambda d:d[1]['year'], reverse=False))

    # print("按年份统计fos信息")
    year_fos = dict()
    for title, info in title_year_fos.items():
        if info['year'] in year_fos:
            year_fos[info['year']] = list(set(year_fos[info['year']]) | set([fos['name'].lower() for fos in info['fos']]))
        else:
            year_fos[info['year']] = [fos['name'].lower() for fos in info['fos']]
    year_fos = dict(sorted(year_fos.items(), key=lambda d:d[0], reverse=False))

    # print("补全信息")
    year = list(year_fos.keys())
    for y in range(year[0], 2017):
        if y not in year_fos.keys():
            t_year = find_smaller_year(y, year)
            year_fos[y] = year_fos[t_year]

    year_fos = dict(sorted(year_fos.items(), key=lambda d: d[0], reverse=False))

    # print("合并信息")
    year = list(year_fos.keys())
    for y, fos in year_fos.items():
        if y == year[0]:
            temp = fos
        else:
            year_fos[y] = list(set(fos) | set(temp))
            temp = year_fos[y]

    temp = dict()
    temp["id"] = sid
    temp["name"] = sid_name[sid]
    temp["fos"] = year_fos
    # print("写入文件")
    out = open("data/seed_scholars_fos.txt", "a")
    out.write(json.dumps(temp) + '\n')
    out.close()


def read_seed_scholar_id_name():
    id_name = dict()
    with open(r"data/seed_scholars_info.txt", "r") as scholars:
        for scholar in scholars:
            scholar_json = json.loads(scholar)
            id_name[scholar_json["id"]] = scholar_json["name"]
    return id_name


if __name__ == '__main__':
    print("读入fos")
    title_fos = read_title_fos()
    print("读入id name")
    sid_name = read_seed_scholar_id_name()
    print("分析")
    i = 0
    for sid, sname in sid_name.items():
        i += 1
        scholar_fos(sid)
        print(str(i) + '/' + str(len(sid_name.keys())))
