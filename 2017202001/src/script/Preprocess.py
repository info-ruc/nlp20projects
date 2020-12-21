#!/usr/bin/env python
# coding: utf-8

# In[25]:


import json
import cv2
import matplotlib.pyplot as plt
import os
import pickle


# In[22]:

'''
train_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/train2014'
val_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/val2014'
cap_path = '/ssd500g/kirlin/repos/self-critical.pytorch/data/dataset_coco.json'
vocab_path = 'vocabulary.pkl'
data_path = 'data.pkl'
'''

class Vocabulary(object):
    '''建立vocabulary类，方便以后word<->index的查找'''
    def __init__(self):
        self.word2ind = dict()
        self.ind2word = dict()
        self.ind = 0         #记录最后当前所有词的最大index
        
    def add_word(self, word):
        if word not in self.word2ind.keys():
            self.word2ind[word] = self.ind
            self.ind2word[ind] = word
            self.ind += 1
    
    def __call__(self, word):
        if word not in self.word2ind.keys():
            return self.word2ind['<ukn>']
        return self.word2ind[word]
    
    def __len__(self):
        return len(self.word2ind)


def read_caption(cap_path):
    with open(cap_path,'r') as load_f:
        load_dict = json.load(load_f)
    new_dict = {}
    for i in range(len(load_dict['images'])):
        if 'val' not in load_dict['images'][i]['filename']:
            new_dict[load_dict['images'][i]['filename']] = [load_dict['images'][i]['sentences'][j]['tokens'] for j in range(5)]
    return new_dict


# In[5]:


def read_images(train_img,val_img):
    train = {}
    val = {}
    '''训练集'''
    List = os.listdir(train_img) #列出文件夹下所有的目录与文件
    for i in range(0,len(List)):
        path = os.path.join(train_img,List[i])
        if os.path.isfile(path):        
            train[List[i]] = cv2.imread(path)
            
    '''验证集'''
    List = os.listdir(val_img)
    for i in range(0,len(List)):
        path = os.path.join(val_img,List[i])
        if os.path.isfile(path):        
            val[List[i]] = cv2.imread(path)
    return train,val


# In[18]:


def One_hot(cap_value,th = 5):  #字典的values,出现次数小于5的词去掉
    '''只有训练集的caption加入这里，验证集的不能加入，未见过的词为<ukn>'''
    cnt_dict = {}
    #统计词频
    for cap in cap_value:
        for i in range(5):
            for word in cap[i]: #已经分词分好的字幕，所以是个列表
                if word not in cnt_dict.keys():
                    cnt_dict[word] = 1
                else:
                    cnt_dict[word] += 1
            
    word2ind = {}
    ind2word = {}
    ind = 0
    for cap in cap_value:
        for i in range(5):
            for word in cap[i]: #已经分词分好的字幕，所以是个列表
                if word not in word2ind.keys() and cnt_dict[word] > th:
                    ind += 1     #1 base，0为padding
                    word2ind[word] = ind
                    ind2word[ind] = word
                    
    
    #将未见过的词，padding，开始符号，结束符号也加入字典里
    word2ind['<pad>'] = 0
    ind2word[0] = '<pad>'
    word2ind['<start>'] = ind + 1
    ind2word[ind+1] = '<start>'
    word2ind['<end>'] = ind + 2
    ind2word[ind+2] = '<end>'
    word2ind['<ukn>'] = ind + 3
    ind2word[ind+3] = '<ukn>'
    
    return word2ind,ind2word
    


# In[7]:


def cap2num(word2ind,caption):
    '''将英文字幕转化为one-hot向量'''
    print(type(caption))
    l = len(word2ind)
    num_dict = {}
    for key in caption: #key是图片名
        num_dict[key] = []
        for j in range(5):
            num_dict[key].append([])
            for word in caption[key][j]:
                arr = [0 for k in range(l)]
                if word in word2ind.keys():
                    arr[word2ind[word]] = 1
                else:
                    arr[word2ind['<ukn>']] = 1
                num_dict[key][j].append(arr)
   
    return num_dict
                        


# In[8]:


def collect(captions):
    '''
    将图片名与字幕统一起来，形成一个tuple
    '''
    data = []
    for key in captions.keys():
        for i in range(5):
            data.append((key,captions[key][i]))
    return data


# In[23]:


def preprocess(cap_path,vocab_path,data_path):
    '''读入字幕'''
    #train_img,val_img = read_images(train_img,val_img)
    caption = read_caption(cap_path)
    #print('reading done!')
    
    '''生成one-hot字典'''
    word2ind,ind2word = One_hot(caption.values(),th = 5) 
    vocab = Vocabulary()
    vocab.word2ind = word2ind
    vocab.ind2word = ind2word
    vocab.ind = len(word2ind)
    
    #不应该现在就转化，否则内存不够
    #caption = cap2num(word2ind,caption)  #将word转换为one-hot向量,每个字幕为一个二维列表，因此长度可变
    #print('one-hot done!')
    #print(caption['COCO_val2014_000000391895.jpg'])
    
    '''合并字幕和图片，并转化成tuple格式'''
    data = collect(caption)
    #print('collect done!')
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
        
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    





