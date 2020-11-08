#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import Preprocess



train_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/train2014'
val_img = '/ssd500g/kirlin/repos/self-critical.pytorch/data/ImageRoot/val2014'
caption = '/ssd500g/kirlin/repos/self-critical.pytorch/data/dataset_coco.json'



class My_Dataset(data.Dataset):
    def __init__(self, img_root, captions, word2ind, transform=None):
        self.img_root = img_root
        self.vocab = word2ind
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
        cap.append(vocab['<start>'])
        for word in caption:
            if word in vocab.keys():
                cap.append(vocab[word])
            else:
                cap.append(vocab['<ukn>'])
        cap.append(vocab['<end>'])
        target = torch.Tensor(cap)
        print(caption)
        return image, target

    def __len__(self):
        return len(self.captions)



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths




def get_loader(train_img, captions, word2ind, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    dataset = My_Dataset(img_root=train_img,
                       captions=captions,
                       word2ind=word2ind,
                       transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


#Data,word2ind,ind2word = Preprocess.preprocess(caption,train_img,val_img)
#data_loader = get_loader(train_img, Data, word2ind, None, 16, True, 2)



