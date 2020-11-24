import pandas as pd
from pandas import Series, DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
from KGQA.standford_check_name import check

# 获取所有的文件
def getfilelist(root_path):
    file_path_list=[]
    file_name=[]
    walk = os.walk(root_path)
    for root, dirs, files in walk:
        for name in files:
            filepath = os.path.join(root, name)
            file_name.append(name)
            file_path_list.append(filepath)
    # print(file_name)
    # print(file_path_list)
    # print(len(file_path_list))
    return file_path_list


class Question_classify():
    def __init__(self):
        # 读取训练数据
        self.train_x,self.train_y=self.read_train_data()
        # 训练模型
        self.model=self.train_model_NB()
    # 获取训练数据
    def read_train_data(self):
        train_x=[]
        train_y=[]
        #file_list=getfilelist("./question/question/")
        # 遍历所有文件
        #for one_file in file_list:
        with(open('../question/question_classification.txt',"r",encoding="utf-8")) as fr:
            data_list=fr.readlines()
            for one_line in data_list:
                label_num = one_line.split('\\t')[1].strip()
                word_list = one_line.split('\\t')[0].strip()
                # 将这一行加入结果集
                train_x.append(word_list)
                train_y.append(label_num)
        return train_x,train_y

    # 训练并测试模型-NB
    def train_model_NB(self):
        X_train, y_train = self.train_x, self.train_y
        self.tv = TfidfVectorizer()

        train_data = self.tv.fit_transform(X_train).toarray()
        clf = MultinomialNB(alpha=0.01)
        clf.fit(train_data, y_train)
        return clf

    # 预测
    def predict(self,question):
        question = [question]
        test_data=self.tv.transform(question).toarray()
        y_predict = self.model.predict(test_data)[0]
        # print("question type:",y_predict)
        return y_predict


if __name__ == '__main__':
    qc=Question_classify()
    question = 'Which David Williams and Stefan Mayer have'
    question = check.check_name(question)
    print(question)
    concept_wds = [i.strip() for i in open('../raw_data/concept.txt', encoding='utf-8') if i.strip()]    #  做一下概念替换便于分类
    for w in concept_wds:
        if w in question:
            question = question.replace(w,'concepts')
    print(qc.predict(question))