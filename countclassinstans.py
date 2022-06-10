
from Molitorutils import *
from torch.utils.data import Dataset
import numpy as np 



img_dir = './dataset/newtotal/imagetest/'
annotations_dir = './dataset/newtotal/annottest/'

annotations_dir ='./dataset/newtotal/EssentialAnnot/'
img_dir ='./dataset/newtotal/EssentialImage/'


annotations_dir ='./dataset/testset/Annotations/'
img_dir ='./dataset/testset/JPEGImages/'

annotations_dir ='./dataset/testset/testannot/'
img_dir ='./dataset/testset/testimages/'

annotations_dir ='./dataset/final/Annotations/'
img_dir ='./dataset/final/JPEGImages/'

annotations_dir ='./dataset/final/Annotations1600/'
img_dir ='./dataset/final/JPEGImages1600/'


annotations_dir ='./dataset/final/AnnotationsPed/'
img_dir ='./dataset/final/JPEGImagesPed/'

dataset1= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area = 700)
dataset2= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area = 1000)
dataset3= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area = 1300)
dataset4= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area = 1600)

ci1 = count_classinstances(dataset1)
ci2 = count_classinstances(dataset2)
ci3 = count_classinstances(dataset3)
ci4 = count_classinstances(dataset4)


lst50 = []
lst100 = []
lst200 = []
for idx, cls in enumerate(ci1):
    print("%0.0f %5s %-5.0f %-5.0f %-5.0f %-5.0f" %(idx, ": ", round(cls),ci2[idx],ci3[idx],ci4[idx] ))
    
    if ci1[idx]>=50:
       lst50.append(idx)
    if ci1[idx]>=100:
       lst100.append(idx)
    if ci1[idx]>=200:
       lst200.append(idx)

print("Threshold 50 signs: " + str(lst50)) 
print("Threshold 100 signs: " + str(lst100)) 
print("Threshold 200 signs: " + str(lst200))


print(dataset1.__len__(),dataset2.__len__(),dataset3.__len__(),dataset4.__len__())

