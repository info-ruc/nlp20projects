import re
from nltk.tag import StanfordNERTagger
import os
import pandas as pd
import nltk


class check(object):
    def __init__(self):
        pass

    def parse_document(document):
        document = re.sub('\n', ' ', document)
        if isinstance(document, str):
            document = document
        else:
            raise ValueError('Document is not string!')
        document = document.strip()
        sentences = nltk.sent_tokenize(document)
        sentences = [sentence.strip() for sentence in sentences]
        return sentences

    def check_name(article_content):
        sentences = check.parse_document(article_content)

        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

        # set java path in environment variables
        #java_path = r'C:\Program Files\Java\jdk1.8.0_161\bin\java.exe'
        #os.environ['JAVAHOME'] = java_path
        # load stanford NER
        sn = StanfordNERTagger('D://stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz',
                               path_to_jar='D://stanford-ner-2018-10-16/stanford-ner.jar')

        # tag sentences   最重要的一步分类算法
        ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]
        # extract named entities
        named_entities = []
        for sentence in ne_annotated_sentences:
            temp_entity_name = ''
            temp_named_entity = None
            for term, tag in sentence:
                # get terms with NE tags
                if tag != 'O':
                    temp_entity_name = ' '.join([temp_entity_name, term]).strip()  # get NE name
                    temp_named_entity = (temp_entity_name, tag)  # get NE and its category
                else:
                    if temp_named_entity:
                        named_entities.append(temp_named_entity)
                        temp_entity_name = ''
                        temp_named_entity = None

        # get unique named entities
        named_entities = list(set(named_entities))
        ###########      named_entities是识别结果      ##########
        name = []
        string = str(article_content)
        for n in named_entities:
            if n[1] == 'PERSON':
                string = string.replace(n[0], 'nr')
            if n[1] == 'ORGANIZATION':
                string = string.replace(n[0], 'nt')

        return string

'''
ttt = 'The case FIFA was prosecuted by Trial Attorney Joseph Palazzo of the Money Laundering and Asset Recovery Section and Assistant U.S. Attorneys Thomas A. Gillice, Luke Jones, Karen Seifert and Deborah Curtis and Special Assistant U.S. Attorney Jacqueline L. Barkett of the U.S. Attorney’s Office for the District of Columbia.'
name = check.check_name(ttt)
print(name)
'''

