import torch
from Molitorutils import *

annot_to ='./dataset/final/Annotations1600/'
img_to ='./dataset/final/JPEGImages1600/'
dataset= pascalVoc(img_to, annot_to, transform=get_transform(train=False))

image, targets = dataset.__getitem__(700)

plotBB(image,targets)
