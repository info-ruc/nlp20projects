#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# In[2]:


class Encoder(nn.Module):
    """Encoder为CNN，具体模型为ResNet"""
    def __init__(self, embedding_size = 512, pooling_kernel = 10):
        """加载预训练模型ResNet-152，同时去除ResNet的最后一层"""
        super(Encoder, self).__init__()
        
        resnet = models.resnet152(pretrained = True)
        modules = list(resnet.children())[:-1]    #去除最后一层fc
        
        self.resnet = nn.Sequential(*modules)
        self.pooling = nn.AvgPool2d(pooling_kernel)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, momentum = 0.01)
        
    def forward(self, images):
        """使用ResNet提取图片特征"""
        #images : [batch_size, channel, resize, resize] ==> [128, 3, 512, 512]
        with torch.no_grad():    #使features不参与导数的计算
            features = self.resnet(images)                  #features: [128, 2048, 10, 10]
        
        features = self.pooling(features)                   #features: [128, 2048, 1, 1]
        features = features.reshape(features.size(0),-1)    #features: [128, 2048]
        features = self.bn(self.linear(features))           #features: [128, 256]
        
        return features
    


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, layer_num = 1, max_seq_len = 20):
        """
        args:
        embedding_size: 转化成词向量的长度，LSTM的输入向量维度
        hidden_size: 隐藏向量h的维度，也就是LSTM每个词的输出维度
        vocab_size: 数据集里出现词的总数+3(<start>,<end>,<ukn>三个特殊符号),
                    用于最后生成词典大小的向量并寻找最大值作为输出
        layer_num: LSTM堆叠的层数，默认为1
        max_seq_len: LSTM输出字符串的最大长度，默认为20
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        #batch_first = True时，输出结果为(batch_size, seq_length, feature)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layer_num, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_len = max_seq_len
        
    def forward(self, features, captions, lengths):
        '''解码CNN Encoder 并生成caption'''
        embeds = self.embedding(captions)
        embeds = torch.cat((features.unsqueeze(1),embeds),1)
        
        #将padding后的序列压缩成一个紧实的序列
        packed = pack_padded_sequence(embeds, lengths, batch_first = True)
        
        hiddens,_ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    def sample(self, features, states=None):
        '''inference阶段,使用贪心算法,给定一个image feature生成一句caption'''
        index = []
        features = features.unsqueeze(1)
        for i in range(self.max_seq_len):
            '''输入lstm和linear后，找到最大值作为第i个词的输出'''
            hiddens, states = self.lstm(features, states)   #hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))     #outputs: (batch_size, vaocab_size)
            _, pred = outputs.max(1)                        #pred: (batch_size)
            
            '''保存至index列表，并将该词作为下一轮的输入'''
            index.append(pred)
            features = self.embedding(pred)                 #features: (batch_size, embed_size)
            features = features.unsqueeze(1)                #features: (batch_size, 1, embed_size)
        
        '''
        将多个tensor的列表变为一个tensor，并将index维度从(max_seq_len, batch_size)
        变为(batch_size, max_seq_len)
        '''
        index = torch.stack(index, 1)    
        
        return index



