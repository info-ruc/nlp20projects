from neo_db.config import graph, CA_LIST, similar_words
import codecs
import os
import json
import base64


def concept():
    data = list(graph.run(
            "MATCH (n:CONCEPT) RETURN n LIMIT 10000"))
    f = open('../raw_data/concept.txt', 'w', encoding='utf-8')
    for d in data:
        f.write(d['n']['conceptName']+'\n')


def author():
    data = list(graph.run(
                "MATCH (n:AUTHOR) RETURN n LIMIT 10000"))
    f = open('../raw_data/author.txt', 'w', encoding='utf-8')
    for d in data:
        f.write(d['n']['authorName']+'\n')
