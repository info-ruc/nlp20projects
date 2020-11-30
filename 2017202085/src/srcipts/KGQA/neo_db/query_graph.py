from neo_db.config import graph, CA_LIST, similar_words
from spider.show_profile import get_profile
import codecs
import os
import json
import base64

def query_coauthor(name,level):

    if level == '1':
        data = graph.run(
        "match(p )-[r:COAUTHOR]->(n:AUTHOR{authorName:'%s'}) return p.authorName, r, n.authorName\
            Union all\
        match(p:AUTHOR {authorName:'%s'}) -[r:COAUTHOR]->(n) return p.authorName, r, n.authorName" % (name,name)
        )
    elif level == '2':
        data = graph.run(
            "match(p:AUTHOR{authorName:'%s'} )-[r:COAUTHOR*1..2]->(n) return p.authorName, r, n.authorName" % (name)
        )
    elif level == '3':
        data = graph.run(
            "match(p:AUTHOR{authorName:'%s'} )-[r:COAUTHOR*1..3]->(n) return p.authorName, r, n.authorName" % (name)
        )
    data = list(data)
    return get_json_data(data)


def get_json_data(data):
    json_data={'data':[],"links":[]}
    d=[]
    d_categery_dict = {}
    for i in data:
        if i['r']['type'] == 'Collaborate':
            d.append(i['p.authorName'])
            d.append(i['n.authorName'])
            d_categery_dict[i['p.authorName']] = 'AUTHOR'
            d_categery_dict[i['n.authorName']] = 'AUTHOR'
            d=list(set(d))
        if i['r']['type'] == 'interest':
            d.append(i['p.authorName'])
            d.append(i['n.conceptName'])
            d_categery_dict[i['p.authorName']] = 'AUTHOR'
            d_categery_dict[i['n.authorName']] = 'CONCEPT'
            d=list(set(d))
        if i['r']['type'] == 'belong2':
            d.append(i['p.authorName'])
            d.append(i['n.affiliationName'])
            d_categery_dict[i['p.authorName']] = 'AUTHOR'
            d_categery_dict[i['n.affiliationName']] = 'AFFILIATION'
            d=list(set(d))
        if i['r']['type'] == 'own':
            d.append(i['p.authorName'])
            d.append(i['n.paperID'])
            d_categery_dict[i['p.authorName']] = 'AUTHOR'
            d_categery_dict[i['n.paperID']] = 'PAPER'
            d=list(set(d))
        if i['r']['type'] == 'refer':
            d.append(i['p.paperID'])
            d.append(i['n.paperID'])
            d_categery_dict[i['p.paperID']] = 'PAPER'
            d_categery_dict[i['n.paperID']] = 'PAPER'
            d=list(set(d))
    name_dict={}
    count=0
    for j in d:
        data_item={}
        name_dict[j]=count
        count+=1
        data_item['name']=j
        data_item['category']=d_categery_dict[j]
        json_data['data'].append(data_item)

    for i in data:
        link_item = {}
        if i['r']['type'] == 'Collaborate':
            link_item['source'] = name_dict[i['p.authorName']]
            link_item['target'] = name_dict[i['n.authorName']]
            link_item['value'] = 'COAUTHOR'
            json_data['links'].append(link_item)
        if i['r']['type'] == 'interest':
            link_item['source'] = name_dict[i['p.authorName']]
            link_item['target'] = name_dict[i['n.conceptName']]
            link_item['value'] = 'AUTHOR2CONCEPT'
            json_data['links'].append(link_item)
        if i['r']['type'] == 'belong2':
            link_item['source'] = name_dict[i['p.authorName']]
            link_item['target'] = name_dict[i['n.affiliationName']]
            link_item['value'] = 'AUTHOR2AFFILIATION'
            json_data['links'].append(link_item)
        if i['r']['type'] == 'own':
            link_item['source'] = name_dict[i['p.authorName']]
            link_item['target'] = name_dict[i['n.paperID']]
            link_item['value'] = 'AUTHOR2PAPER'
            json_data['links'].append(link_item)
        if i['r']['type'] == 'refer':
            link_item['source'] = name_dict[i['p.paperID']]
            link_item['target'] = name_dict[i['n.paperID']]
            link_item['value'] = 'CITATION'
            json_data['links'].append(link_item)

    return json_data


def query_author_info(name):
    data = graph.run("match(p :AUTHOR{authorName:'%s'})-[r]->(n) return p, r, n "%(name))
    data = list(data)
    json_data = {'data': [], "links": []}

    data_item1 = {}
    data_item1['name'] = name
    data_item1['category'] = "AUTHOR"
    json_data['data'].append(data_item1)
    count = 0
    name_dict = {}
    name_dict[data_item1['name']] = count
    count+=1
    for i in data:
        if i['r']['type'] == 'interest':
            data_item2 = {}
            data_item2['name'] = i['n']['conceptName']
            data_item2['category'] = "CONCEPT"
            name_dict[data_item2['name']] = count
            count += 1
            link_item = {}
            link_item['source'] = name_dict[i['p']['authorName']]
            link_item['target'] = name_dict[i['n']['conceptName']]
            link_item['value'] = 'AUTHOR2CONCEPT'
            json_data['data'].append(data_item2)
            json_data['links'].append(link_item)
        if i['r']['type'] == 'belong2':
            data_item2 = {}
            data_item2['name'] = i['n']['affiliationName']
            data_item2['category'] = "AFFILIATION"
            name_dict[data_item2['name']] = count
            count += 1
            link_item = {}
            link_item['source'] = name_dict[i['p']['authorName']]
            link_item['target'] = name_dict[i['n']['affiliationName']]
            link_item['value'] = 'AUTHOR2AFFILIATION'
            json_data['data'].append(data_item2)
            json_data['links'].append(link_item)
        if i['r']['type'] == 'own':
            data_item2 = {}
            data_item2['name'] = i['n']['paperID']
            data_item2['category'] = "PAPER"
            name_dict[data_item2['name']] = count
            count += 1
            link_item = {}
            link_item['source'] = name_dict[i['p']['authorName']]
            link_item['target'] = name_dict[i['n']['paperID']]
            link_item['value'] = 'AUTHOR2PAPER'
            json_data['data'].append(data_item2)
            json_data['links'].append(link_item)
        if i['r']['type'] == 'Collaborate':
            data_item2 = {}
            data_item2['name'] = i['n']['authorName']
            data_item2['category'] = "AUTHOR"
            name_dict[data_item2['name']] = count
            count += 1
            link_item = {}
            link_item['source'] = name_dict[i['p']['authorName']]
            link_item['target'] = name_dict[i['n']['authorName']]
            link_item['value'] = 'COAUTHOR'
            json_data['data'].append(data_item2)
            json_data['links'].append(link_item)

    return json_data


# f = codecs.open('./static/test_data.json','w','utf-8')
# f.write(json.dumps(json_data,  ensure_ascii=False))
def get_KGQA_answer(sqls):
    json_data = {'data': [], "links": []}
    for sql_ in sqls:
        question_type = sql_['question_type']
        queries = sql_['sql']
        for query in queries:
            data = graph.run(query)
            data = list(data)
            res = get_json_data(data)
            json_data['data'].extend(res['data'])
            json_data['links'].extend(res['links'])
    return json_data


def query_paper_info(id):
    json_data = {'data': [], "links": []}
    data_item1 = {}
    data_item1['name'] = id
    data_item1['category'] = "AUTHOR"
    json_data['data'].append(data_item1)
    count = 0
    name_dict = {}
    name_dict[data_item1['name']] = count
    count += 1
    data1 = graph.run("match(p:PAPER{paperID:'%s'} )-[r]->(n:PAPER) return p, r, n" % (id))
    data1 = list(data1)
    for i in data1:
        data_item2 = {}
        data_item2['name'] = i['n']['paperID']
        data_item2['category'] = "PAPER"
        name_dict[data_item2['name']] = count
        count += 1
        link_item = {}
        link_item['source'] = name_dict[i['p']['paperID']]
        link_item['target'] = name_dict[i['n']['paperID']]
        link_item['value'] = 'CITATION'
        json_data['data'].append(data_item2)
        json_data['links'].append(link_item)
    data2 = graph.run("match(p:AUTHOR )-[r:AUTHOR2PAPER]->(n:PAPER{paperID:'%s'}) return p, r, n"%(id))
    data2 = list(data2)
    for i in data2:
        data_item2 = {}
        data_item2['name'] = i['p']['authorName']
        data_item2['category'] = "AUTHOR"
        name_dict[data_item2['name']] = count
        count += 1
        link_item = {}
        link_item['source'] = name_dict[i['p']['authorName']]
        link_item['target'] = name_dict[i['n']['paperID']]
        link_item['value'] = 'AUTHOR2PAPER'
        json_data['data'].append(data_item2)
        json_data['links'].append(link_item)
    data3 = graph.run(
        "match(p:PAPER)-[r]->(n:PAPER{paperID:'%s'} ) return p, r, n " % (id))
    data3 = list(data3)
    for i in data3:
        data_item2 = {}
        data_item2['name'] = i['p']['paperID']
        data_item2['category'] = "PAPER"
        name_dict[data_item2['name']] = count
        count += 1
        link_item = {}
        link_item['source'] = name_dict[i['p']['paperID']]
        link_item['target'] = name_dict[i['n']['paperID']]
        link_item['value'] = 'CITATION'
        json_data['data'].append(data_item2)
        json_data['links'].append(link_item)

    return json_data


def get_hot_author():
    data = graph.run(
        "match(p:AUTHOR{authorName:'%s'} )-[r:COAUTHOR*1..3]->(n) return p.authorName, n.authorName"
    )
    data = list(data)



