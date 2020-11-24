#!/usr/bin/env python3
# coding: utf-8
# File: question_classifier.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-4

import os
import ahocorasick
from nltk.corpus import wordnet as wn

class QuestionClassifier:
    def __init__(self):
        # 加载特征词
        self.author_wds= [i.strip() for i in open('./raw_data/author.txt', encoding='utf-8') if i.strip()]
        self.concept_wds= [i.strip() for i in open('./raw_data/concept.txt', encoding='utf-8') if i.strip()]
        self.region_words = set(self.author_wds + self.concept_wds)
        self.deny_words = [i.strip() for i in open('./raw_data/deny.txt', encoding='utf-8') if i.strip()]
        # 构造领域actree
        self.region_tree = self.build_actree(list(self.region_words))
        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()
        # 问句疑问词
        self.author2paper_qwds = ['publish', 'report', 'deliver', 'issue', 'announce', 'publishes', 'reports', 'delivers', 'issues', 'announces','published', 'reported', 'delivered', 'issued', 'announced','write','writes','writed','wrote']
        self.coauthor_qwds = ['cooperate', 'collaborate', 'work together', 'cooperates', 'collaborates', 'works together']
        self.author2concept_qwds = ['interest', 'research', 'study', 'interests', 'researches', 'studies']
        self.author2affliation_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现', '伴随发生', '伴随', '共现']

        print('model init finished ......')

        return

    '''分类主函数'''
    def classify(self, question):
        data = {}
        medical_dict = self.check_medical(question)
        if not medical_dict:
            return {}
        data['args'] = medical_dict
        #收集问句当中所涉及到的实体类型
        types = []
        for type_ in medical_dict.values():
            types += type_
        question_type = 'others'

        question_types = []

        if self.check_words(self.author2paper_qwds, question) and ('author' in types):
            question_type = 'author2paper'
            question_types.append(question_type)

        if self.check_words(self.coauthor_qwds, question) and ('author' in types):
            question_type = 'coauthor'
            question_types.append(question_type)

        if self.check_words(self.author2concept_qwds, question) and ('concept' in types):
            question_type = 'author2concept'
            question_types.append(question_type)

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''构造词对应的类型'''
    def build_wdtype_dict(self):
        wd_dict = dict()
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.author_wds:
                wd_dict[wd].append('author')
            if wd in self.concept_wds:
                wd_dict[wd].append('concept')
        return wd_dict

    '''构造actree，加速过滤'''
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    '''问句过滤'''
    def check_medical(self, question):
        region_wds = []
        for i in self.region_tree.iter(question):
            wd = i[1][1]
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        final_dict = {i:self.wdtype_dict.get(i) for i in final_wds}

        return final_dict

    '''基于特征词进行分类'''
    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False


if __name__ == '__main__':
    handler = QuestionClassifier()
    while 1:
        question = input('input an question:')
        data = handler.classify(question)
        print(data)