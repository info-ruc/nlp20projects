#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from torchvision import transforms
from Preprocess import Vocabulary   
from network import Encoder, Decoder
from PIL import Image
import yaml


# In[4]:


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(os.environ['CUDA_VISIBLE_DEVICES'])


# In[5]:


def load_image(image_path, transform = None, size = (512,512)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.LANCZOS)   #使用Image.LANCZOS进行重采样
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
        
    return image


# In[6]:


def infer(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    
    #加载词汇表
    with open(args['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
    with open(args['data_path'], 'rb') as f:
        Data = pickle.load(f)
        
    #在测试阶段使用model.eval()，将BN和Dropout固定,使用训练好的值
    encoder = Encoder(args['embed_size'], args['pooling_kernel']).eval().cuda()
    decoder = Decoder(args['embed_size'], args['hidden_size'], len(vocab), args['num_layers']).cuda()
    #encoder = encoder.cuda()
    #decoder = decoder.cuda()
    
    #加载训练时的参数
    encoder.load_state_dict(torch.load(args['encoder_path']))
    decoder.load_state_dict(torch.load(args['decoder_path']))
    
    #加载图片
    image = load_image(args['val_img_path'], transform, (args['resize'],args['resize']))
    image_tensor = image.cuda()
    
    #送入模型并输出caption
    feature = encoder(image_tensor)
    index = decoder.sample(feature)
    index = index[0].cpu().numpy()
    
    #将index转化成word
    words = []
    for ind in index:
        word = vocab.idx2word[word_id]
        words.append(word)
        if word == '<end>':
            break
    
    sentence = ' '.join(words[1:-1])    #去掉开头和结尾的特殊字符<start>,<end>
    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    


# In[8]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type = str, default = 'show_and_tell.yaml')
    '''
    parser.add_argument('--resize', type = int, default = 512)
    
    #数据，模型路径
    parser.add_argument('--val_image_path', type = str, default = '')
    parser.add_argument('--encoder_path', type = str, default = '')
    parser.add_argument('--decoder_path', type = str, default = '')
    parser.add_argument('--vocab_path', type = str, default = '')
    
    #模型参数，和train.py文件应一致
    parser.add_argument('--embed_size', type = int, default = 256)
    parser.add_argument('--hidden_size', type = int, default = 512)
    parser.add_argument('--num_layers', type = int, default = 1)
    '''
    
    args = parser.parse_args()
    with open(args.c, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
        print(cfgs)
        
    infer(cfgs)


# In[ ]:




