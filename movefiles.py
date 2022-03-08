
import os
from Molitorutils import *
import torchvision.transforms as T 
from torchvision.utils import save_image
import numpy as np
import shutil
from lxml import etree as ET
import torchvision.transforms.functional as TF
import random


def Movefile1():
    rootannot = "./data/odden/Annotations/"
    rootmoveto = "./data/odden/JPEGImages/"
    rootmovefrom = "./data/odden/part1/"

    filenames = sorted(os.listdir(rootannot))

    for idx, file in enumerate(filenames):
        filename = file[:-4] + ".jpg"
        
        os.replace(rootmovefrom+filename, rootmoveto + filename)
    

def cropandequalizeImage(image):
    heightOrignial = image.shape[1]
    width = image.shape[2]
    height = int(heightOrignial*0.5)
    top = int(heightOrignial*0.2)
    image = T.functional.crop(image,top,0,height,width )
    image *= 255
    image = torch.tensor(image,dtype=torch.uint8)
    image = T.functional.equalize(image)
    image = image/255
    return image


def preprocessImages(dataset):
    
    #from 
    rootannotfrom = "./data/totaldataset/marieberg/Annotations/"
    imagesrootmovefrom = './data/totaldataset/marieberg/JPEGImages/'
    # to 
    rootannotto = "./data/agumenteddata/croped/Annotations/"
    imagerootmoveto = "./data/agumenteddata/croped/JPEGImages/"
    
    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        print(idx)
        print(filenames[idx])
        #if np.in1d(cls,targets['labels']).any():#cls.any() in targets['labels'].any():
        image = T.Grayscale()(image)
        image = cropandequalizeImage(image)        
        save_image(image,imagerootmoveto + imagenames[idx])
        
        tree = ET.parse(os.path.join(rootannotfrom,filenames[idx]))
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                c4 = child[4]
                c4[1].text = str(int(float(c4[1].text)-800))
                c4[3].text = str(int(float(c4[3].text)-800))
        tree.write(rootannotto + filenames[idx])

        #move annot
        #shutil.copyfile(rootannotfrom+filenames[idx], rootannotto + filenames[idx])
        
def sortImages(cls,dataset):
    cls2 = cls
    cls = cls +1
    #from 
    rootannotfrom ="./data/agumenteddata/croped/Annotations/"
    imagesrootmovefrom = "./data/agumenteddata/croped/JPEGImages/"
    # to 

    rootannotto = "./data/agumenteddata/"+str(cls2)+"/Annotations/"
    imagerootmoveto = "./data/agumenteddata/"+str(cls2)+"/JPEGImages/"
    
    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        print(idx)
        if np.in1d(cls,targets['labels']).any():
            print("Moving: " + filenames[idx])
            #save image
            shutil.copyfile(imagesrootmovefrom+imagenames[idx], imagerootmoveto + imagenames[idx])
            
            #move annot
            shutil.copyfile(rootannotfrom+filenames[idx], rootannotto + filenames[idx])
            


def augmentImages(dataset,cls,rootannotfrom,imagesrootmovefrom):
    cls = cls +1

    # to 
    rootannotto = "./data/agumenteddata/"+str(cls-1)+"/AugAnnotations/"
    imagerootmoveto = "./data/agumenteddata/"+str(cls-1)+"/AugImages/"
    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    
    #classinstansces = count_classinstances(dataset)
    # for idx, cls in enumerate(classinstansces):
    #     print(idx, cls)


    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        boxes = targets['boxes']
        print(idx)
        for idx2, box in enumerate(boxes):

            
            if targets['labels'][idx2] != cls:

                i = int(box[1])
                j = int(box[0])
                h = int(box[3]-box[1])
                w = int(box[2]-box[0])
                v = torch.tensor(0)
                image = T.functional.erase(image,i,j,h,w,v)
                
        tree = ET.parse(os.path.join(rootannotfrom,filenames[idx]))
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                label = child[0].text

                if int(label) !=(cls-1):
                    parent = child.getparent()
                    parent.remove(child)
        
        
        
        org_image= image
        # rotator = T.RandomAdjustSharpness(2,1)
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_sharpness.jpg")
        # tree.write(rootannotto +"AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_sharpness.xml")
        
        
        rotator = T.RandomPerspective(0.25,1)
        image = rotator(org_image)
        save_image(image,imagerootmoveto + "AUGMENTED" +imagenames[idx][:-4]+"_"+str(cls-1)+"_perspective.jpg")
        tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_perspective.xml")
        
        
        # rotator = T.RandomHorizontalFlip(1)
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_horizontal.jpg")
        # tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_horizontal.xml")
        
        
        # rotator = T.RandomRotation(15)
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_rotation.jpg")
        # tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_rotation.xml")
        
        
        
        # rotator = T.GaussianBlur(9, sigma=(0.1, 2.0))
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_blur.jpg")
        # tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_blur.xml")
        
cls = 45 
imagesrootmovefrom = "./data/agumenteddata/"+str(cls)+"/JPEGImages/" #"./data/agumenteddata/croped/JPEGImages/"#
rootannotfrom  = "./data/agumenteddata/"+str(cls)+"/Annotations/" # "./data/agumenteddata/croped/Annotations/"#
dataset= pascalVoc( imagesrootmovefrom,rootannotfrom, transform=get_transform(train=False),area=700)

#sortImages(cls, dataset)
augmentImages(dataset,cls,rootannotfrom,imagesrootmovefrom)

# class nr 

"""
14 0    stop

*2
43 84   no parking         DONE
44 106  no stop or park    
45 138  parking

rotate, random perspective


*4
13 41   give way            DONE
18 38   danger              DONE

rotate, random perspective, horizonflip, sharpness

*10
12 16   priority road       DONE 
22 15   uneven road         DONE

rotate, random perspective, horizonflip, sharpness



*1
1  247  30 
17 191  no entry 
38 174  keep right

/3
27 928  pedestrian cross
"""
