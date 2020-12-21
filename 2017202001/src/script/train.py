#!/usr/bin/env python
# coding: utf-8

# In[6]:


import argparse
import yaml
import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pickle
import numpy as np
from data_loader import get_loader
from Preprocess import Vocabulary, preprocess    
from network import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# In[7]:


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0,1')
#print(os.environ['CUDA_VISIBLE_DEVICES'])

def train(args):
    #数据预处理，生成vocab和data
    preprocess(args['cap_path'],args['vocab_path'],args['data_path'])
    
    if not os.path.exists(args['model_path']):
        os.mkdir(args['model_path'])
        
    #对图片进行处理，进行数据增强
    transform = transforms.Compose([
        transforms.Resize((args['resize'],args['resize'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    with open(args['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
        
    with open(args['data_path'], 'rb') as f:
        Data = pickle.load(f)
        
    data_loader = get_loader(args['train_img_path'], Data, vocab,
                            transform, args['batch_size'],
                            shuffle=True, num_workers=args['num_workers'])
    
    encoder = Encoder(args['embed_size'], args['pooling_kernel']).cuda()
    decoder = Decoder(args['embed_size'], args['hidden_size'], len(vocab), args['num_layers']).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args['learning_rate'])
    
    total_step = len(data_loader)
    for epoch in range(args['num_epochs']):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.cuda()
            captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            
            #打印训练信息
            if i % args['log_step'] == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                     .format(epoch, args['num_epochs'], i, total_step, loss.item(), np.exp(loss.item())))
            
            #保存模型
            if (i+1) % args['save_step'] == 0: 
                torch.save(decoder.state_dict(), os.path.join(
                    args['model_path'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args['model_path'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                
        #每个epoch结束也保存一次模型
        torch.save(decoder.state_dict(), os.path.join(
            args['model_path'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        torch.save(encoder.state_dict(), os.path.join(
            args['model_path'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
    


# In[8]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type = str, default = 'show_and_tell.yaml')
    
    '''
    #路径，数据参数设置
    parser.add_argument('--model_path', type = str, default = 'models')
    parser.add_argument('--crop_size', type = int, default = 224)
    parser.add_argument('--vocab_path', type = str, default = '')
    parser.add_argument('--train_img_path', type = str, default = '')
    parser.add_argument('--data_path', type = str, default = '')
    parser.add_argument('--log_step', type = int, default = 10)
    parser.add_argument('--save_step', type = int, default = 100)
    
    #模型参数
    parser.add_argument('--embed_size', type = int, default = 256)
    parser.add_argument('--hidden_size', type = int, default = 512)
    parser.add_argument('--num_layers', type = int, default = 1)
    
    #训练参数
    parser.add_argument('--num_epochs', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    '''
    args = parser.parse_args()
    
    with open(args.c, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
        print(cfgs)
          
    train(cfgs)
    


# In[ ]:




