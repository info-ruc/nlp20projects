#!/usr/bin/env python
# coding: utf-8


import json
import cv2
import matplotlib.pyplot as plt
import os



train_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/train2014'
val_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/val2014'
caption = '/ssd500g/kirlin/repos/self-critical.pytorch/data/dataset_coco.json'



def read_caption(cap_path):
    '''
    读入字幕
    args:字幕路径，为一个json格式文件
    returns:一个字典，key为对应的图片文件字符串，value为一个列表，长度为5，为5个字幕，即每张图片有5个不同的字幕
    '''
    with open(cap_path,'r') as load_f:
        load_dict = json.load(load_f)
    new_dict = {}
    for i in range(len(load_dict['images'])):
        if 'val' not in load_dict['images'][i]['filename']:
            new_dict[load_dict['images'][i]['filename']] = [load_dict['images'][i]['sentences'][j]['tokens'] for j in range(5)]
    return new_dict



def read_images(train_img,val_img):
    '''
    读图片，由于数据集太大，无法一次性读入内存，此函数暂时不用
    '''
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


def One_hot(cap_value,th = 5):  #字典的values,出现次数小于5的词去掉
    '''
    建立一个word<->index的双向字典
    只有训练集的caption加入这里，验证集的不能加入，未见过的词为<ukn>
    '''
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
                    word2ind[word] = ind
                    ind2word[ind] = word
                    ind += 1
    
    #将未见过的词，开始符号，结束符号也加入字典里
    word2ind['<ukn>'] = ind
    ind2word[ind] = '<ukn>'
    word2ind['<start>'] = ind
    ind2word[ind] = 'ukn'
    word2ind['<end>'] = ind
    ind2word[ind] = 'ukn'
    return word2ind,ind2word
    



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
                        


def collect(captions):
    '''
    将图片名与字幕统一起来，形成一个tuple
    '''
    data = []
    for key in captions.keys():
        for i in range(5):
            data.append((key,captions[key][i]))
    return data



def preprocess(cap_path,train_img,val_img):
    '''读入字幕'''
    #train_img,val_img = read_images(train_img,val_img)
    caption = read_caption(cap_path)
    print('reading done!')
    
    '''生成one-hot字典'''
    word2ind,ind2word = One_hot(caption.values(),th = 5) 
    #不应该现在就转化，否则内存不够
    #caption = cap2num(word2ind,caption)  #将word转换为one-hot向量,每个字幕为一个二维列表，因此长度可变
    print('one-hot done!')
    #print(caption['COCO_val2014_000000391895.jpg'])
    
    '''合并字幕和图片，并转化成tuple格式'''
    data = collect(caption)
    print('collect done!')
    
    return data,word2ind,ind2word
    



#data,d1,d2 = preprocess(caption,train_img,val_img)
#print(data[0])




