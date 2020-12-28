#!/usr/bin/env python
# coding: utf-8


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


def load_image(image_path, transform = None, size = (512,512)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.LANCZOS)   #使用Image.LANCZOS进行重采样
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
        
    return image


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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type = str, default = 'show_and_tell.yaml')
    args = parser.parse_args()

    with open(args.c, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
        print(cfgs)
        
    infer(cfgs)
