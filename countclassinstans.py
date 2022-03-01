
from Molitorutils import *
from torch.utils.data import Dataset
import numpy as np 

class pascalVoc(Dataset):
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


img_dir = './data/totaldataset/JPEGImages/'
annotations_dir = './data/totaldataset/Annotations/'
dataset= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area=700)

classinstansces = count_classinstances(dataset)


img_dir = './data/german dataset/PascalVoc/JPEGImages/'
annotations_dir = './data/german dataset/PascalVoc/Annotations/'
germandataset= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area=700)

germanclassinstansces = count_classinstances(germandataset)

    
swecoclass = np.subtract(classinstansces,germanclassinstansces)
swecolst50 = []
lst50 = []
lst100 = []
lst200 = []
for idx, cls in enumerate(germanclassinstansces):
    print("%0.0f %5s %-5.0f %-5s %-5.0f %-5s %-5.0f" %(idx, " German: ", round(cls), " sweco: ", swecoclass[idx],  " total: ", classinstansces[idx]))
    
    if classinstansces[idx]>=50:
       lst50.append(idx)
    if classinstansces[idx]>=100:
       lst100.append(idx)
    if classinstansces[idx]>=200:
       lst200.append(idx)
    if swecoclass[idx]>=50:
       swecolst50.append(idx)

print("Threshold 50 signs: " + str(lst50)) 
print("Threshold 100 signs: " + str(lst100)) 
print("Threshold 200 signs: " + str(lst200))
print("Threshold 50 Sweco signs: " + str(swecolst50))

print("Percentage Sweco signs with over 50 instanses of all Sweco signs: {:.2f}". format(sum(swecoclass[swecolst50])/sum(swecoclass)*100))

print(dataset.__len__())





