#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from Preprocess import preprocess 
from Preprocess import Vocabulary


# In[6]:


'''
train_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/train2014'
val_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/val2014'
caption = '/ssd500g/kirlin/repos/self-critical.pytorch/data/dataset_coco.json'
vocab_path = 'vocabulary.pkl'
data_path = 'data.pkl'
'''


# In[28]:


class My_Dataset(data.Dataset):
    def __init__(self, img_root, captions, vocab, transform=None):
        self.img_root = img_root
        self.vocab = vocab
        self.captions = captions
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        captions = self.captions
        
        
        caption = captions[index][1]
        img_name = captions[index][0]

        image = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')       
      
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        cap = []
        cap.append(vocab('<start>'))
        for word in caption:
            cap.append(vocab(word))
        cap.append(vocab('<end>'))
        target = torch.Tensor(cap)
        
        return image, target

    def __len__(self):
        return len(self.captions)


# In[29]:


def collate_fn(Data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 512, 512).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 512, 512).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    Data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*Data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


# In[30]:


def get_loader(train_img, captions, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    dataset = My_Dataset(img_root=train_img,
                       captions=captions,
                       vocab=vocab,
                       transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


# In[31]:


'''
#preprocess(caption,train_img,val_img,vocab_path,data_path)
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(data_path, 'rb') as f:
    Data = pickle.load(f)
    
transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0,406),
                           (0.229, 0.224, 0.225))
    ])

data_loader = get_loader(train_img, Data, vocab, transform, 16, True, 2)
'''


# In[32]:


'''
for i, (images, captions, lengths) in enumerate(data_loader):
    print(captions)
    if i > 2:
        break
'''




