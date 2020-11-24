#!/usr/bin/env python3
# coding: utf-8
# File: question_parser.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-4

class QuestionPaser:

    '''构建实体节点'''
    def build_entitydict(self, args):
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)

        return entity_dict

    '''解析主函数'''
    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']
        sqls = []
        for question_type in question_types:
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []
            if question_type == 'author2paper':
                sql = self.sql_transfer(question_type, entity_dict.get('author'))

            elif question_type == 'coauthor':
                sql = self.sql_transfer(question_type, entity_dict.get('author'))

            elif question_type == 'author2concept':
                sql = self.sql_transfer(question_type, entity_dict.get('concept'))

            if sql:
                sql_['sql'] = sql

                sqls.append(sql_)

        return sqls

    '''针对不同的问题，分开进行处理'''
    def sql_transfer(self, question_type, entities):
        if not entities:
            return []

        # 查询语句
        sql = []
        # 查询作者发表论文
        print(entities)
        if question_type == 'author2paper':
            for i in entities:
                sql.append("match(p:AUTHOR{authorName:'%s'} )-[r:AUTHOR2PAPER]->(n) return p.authorName, r, n.paperID"%i )

        # 查询合作者
        elif question_type == 'coauthor':
            sql = ["match(p:AUTHOR{authorName:{0}} )-[r:COAUTHOR]->(n) return p.authorName, r, n.authorName".format(i) for i in entities]

        # 查询作者兴趣
        elif question_type == 'author2concept':
            sql = ["match(p)-[r:AUTHOR2CONCEPT]->(n:concept{conceptName:{0}}) return p.authorName, r, n.conceptName".format(i) for i in entities]

        return sql



if __name__ == '__main__':
    handler = QuestionPaser()
