import requests
import tensorflow as tf


def sort_by_weight(item):
    return item['weight']


def get_knowledge(word):
    obj = requests.get('http://api.conceptnet.io/c/en/' +
                       word+'?filter=/c/en').json()
    related = obj['edges']
    related.sort(key=sort_by_weight, reverse=True)

    scene_info = []
    related_info = []
    for edge in related:
        if edge['end']['language'] != 'en':
            continue
        end_word = edge['end']['label']
        if edge['rel']['label'] == 'AtLocation':
            scene_info.append(end_word)
        else:
            related_info.append(end_word)

    return scene_info, related_info
    # ['rel']['label']
    # ['weight']
