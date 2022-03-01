import torch
import numpy as np 
import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import detection.transforms as T 
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F
import matplotlib
import detection.utils as utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.engine import train_one_epoch, evaluate

class cvatDataset(Dataset):
   def __init__(self, img_dir, annotations_dir, area = 400,takeclass = np.linspace(0,48,49), transform=None, target_transform = None ):
       self.annotation_dir = annotations_dir
       self.img_dir = img_dir
       self.transform = transform
       self.target_transform = target_transform
       self.takeclass = takeclass
      
       self.area = area
      
       self.imagenames = sorted(os.listdir(self.img_dir))
       self.annotnames = sorted(os.listdir(self.annotation_dir))
      
   def __len__(self):
       return len(os.listdir(self.annotation_dir))
 
   def __getitem__(self, idx):
       image = Image.open(self.img_dir +self.imagenames[idx]).convert("RGB")
 
       file = ET.parse(os.path.join(self.annotation_dir,self.annotnames[idx])).getroot()
    
       num_objs = 0
       for child in file:
           if child.tag == 'object':
               c4 = child[4]
               width = int(float(c4[2].text))-int(float(c4[0].text))
               height = int(float(c4[3].text))-int(float(c4[1].text))
               testarea = width*height
               if testarea < self.area or int(child[0].text) not in self.takeclass:
                   continue
               num_objs +=1
       boxes = np.zeros((num_objs,4))
       labels = np.zeros(num_objs)
       i = 0
       for child in file:
           if child.tag == 'object':
               c4 = child[4]
               width = int(float(c4[2].text))-int(float(c4[0].text))
               height = int(float(c4[3].text))-int(float(c4[1].text))
               testarea = width*height
               if testarea < self.area or int(child[0].text) not in self.takeclass:
                   continue
               boxes[i,:] = np.array([int(float(c4[0].text)), int(float(c4[1].text)), int(float(c4[2].text)), int(float(c4[3].text))])
               labels[i] = 1#int(child[0].text)
               i += 1
              
       boxes = torch.as_tensor(boxes,dtype=torch.float32)
       labels = torch.as_tensor(labels,dtype=torch.int64)
       image_id = torch.tensor([idx])
       area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
       iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      
       target = {}
       target["boxes"] = boxes
       target["labels"] = labels
       target["image_id"] = image_id
       target["area"] = area
       target["iscrowd"] = iscrowd
 
       if self.transform is not None:
           image, target = self.transform(image, target)
 
       return image, target
 
 
class germanDataset(Dataset):
   def __init__(self, dir, transform=None, target_transform = None ):
       self.dir = dir
       self.transform = transform
       self.target_transform = target_transform
       self.len = len([f for f in os.listdir(self.dir) if f.endswith('.ppm') ])
       self.filenames = sorted([f for f in os.listdir(self.dir) if f.endswith('.ppm')])
      
       with open(self.dir+"gt.txt") as f:
           lines = f.read().splitlines()
       self.annotations = lines
      
   def __len__(self):
       return self.len
 
   def __getitem__(self, idx):
       filename = self.filenames[idx]
       image = Image.open(self.dir + filename).convert("RGB")
       annotations = [idx for idx in self.annotations if idx[0:9].lower() == filename.lower()]
 
       num_objs = len(annotations)
       boxes = np.zeros((num_objs,4))
       labels = np.zeros(num_objs)
 
       for idx, element in enumerate(annotations):
           info = element.split(";")
           boxes[idx,:] = np.array([info[1],info[2],info[3],info[4]])
           labels[idx] = np.array([info[5]])
              
       boxes = torch.as_tensor(boxes,dtype=torch.float32)
       labels = torch.as_tensor(labels,dtype=torch.int64)
       image_id = torch.tensor([idx])
       area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
       iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      
       target = {}
       target["boxes"] = boxes
       target["labels"] = labels
       target["image_id"] = image_id
       target["area"] = area
       target["iscrowd"] = iscrowd
 
       if self.transform is not None:
           image, target = self.transform(image, target)
 
       return image, target
 
 
def get_transform(train):
   transforms = []
   transforms.append(T.ToTensor())
   if train:
       transforms.append(T.RandomHorizontalFlip(0.5))
   return T.Compose(transforms)
 
 
def show(imgs):
   if not isinstance(imgs, list):
       imgs = [imgs]
   fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
   for i, img in enumerate(imgs):
       img = img.detach()
       img = F.to_pil_image(img)
       axs[0, i].imshow(np.asarray(img))
       axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
   plt.show()
 
def plotBB(image, target,width = 5):
   image *=255
   image = torch.tensor(image,dtype=torch.uint8)
   labels = [str(i) for i in target['labels'].tolist()]
   imageBB = torchvision.utils.draw_bounding_boxes(image,target['boxes'],labels, colors="orange",width = width )
   show(imageBB)
  
class laserdata():
   def __init__(self):
       self.dir = './data/cvat2/Lindhagsterrassen/Laserdata/'
      
       file = open(self.dir + "block1.txt", "r")
       line_count = 0
       for line in file:
           if line != "\n":
               line_count += 1
       file.close()               
       self.len = line_count
      
   def __len__(self):
       return self.len
 
  
   def __getitem__(self, idx):
      
       fp = open(self.dir + "block1.txt", "r")
       for i, line in enumerate(fp):
           if i == idx:
               item = line
               break
              
       fp.close() 
       
       return item
  
def save_laserdata2csv(dataset, filename):
   # open the file in the write mode
   f = open(filename + '.csv', 'w')
 
   # create the csv writer
   writer = csv.writer(f)
 
   for i in range(dataset.len):
       line = dataset.__getitem__(i)
       lst = line.split()   
       # write a row to the csv file
       writer.writerow(lst[0:3])
      
       if i%10000 == 0:
           print(str(i) +"/" +str(dataset.len))
 
       # close the file
 
   f.close()
  

def count_classinstances(dataset):
    classinstansces = np.zeros(48)
    for idx in range(dataset.__len__()):
       file = ET.parse(os.path.join(dataset.annotation_dir,dataset.annotnames[idx])).getroot()
       num_objs = 0

       for child in file:
           if child.tag == 'object':
               c4 = child[4]
               width = int(float(c4[2].text))-int(float(c4[0].text))
               height = int(float(c4[3].text))-int(float(c4[1].text))
               testarea = width*height
               if testarea < dataset.area or int(child[0].text) not in dataset.takeclass:
                   continue
               classinstansces[int(child[0].text)] +=1
    
    
    return classinstansces

def displayone(model, idx, dataset,thres):
    model.eval()
    with torch.no_grad():
        image, targets = dataset.__getitem__(idx)
        image = list([image])
        model = model.to('cpu')
        pred = model(image)
        scores = pred[0]['scores']
        scores_ind = np.argwhere(scores > thres)
        boxes = pred[0]['boxes']
        boxes = boxes[scores_ind]
        image = torch.mul(image[0],255)
        image = torch.tensor(image,dtype=torch.uint8)
        scr = [str(score) for score in scores]
        print(scores)
        imageBB = torchvision.utils.draw_bounding_boxes(image,boxes[0],labels=scr, colors='orange', width= 4)
        show(imageBB)