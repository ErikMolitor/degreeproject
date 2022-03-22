import os
from tkinter import image_names

from matplotlib.pyplot import bar_label
from Molitorutils import *
import torchvision.transforms as T 
from torchvision.utils import save_image
import numpy as np
import shutil
from lxml import etree as ET
import torchvision.transforms.functional as TF
import random
import pandas as pd

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

def cropImage(image):
    heightOrignial = image.shape[1]
    width = image.shape[2]
    height = int(heightOrignial*0.5)
    top = int(heightOrignial*0.2)
    image = T.functional.crop(image,top,0,height,width )
    return image

def disnonannotimage(dataset, img_dir, annot_dir, img_to, annot_to):
    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    i = 0
    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        print(idx)


        if len(targets['labels']) == 0:
            print("removing: ")
            i += 1
            #save image
            os.remove(img_dir+imagenames[idx])
            #shutil.copyfile(img_dir+imagenames[idx], img_to + imagenames[idx])
            
            #move annot
            os.remove(annot_dir+filenames[idx])
            #shutil.copyfile(annot_dir+filenames[idx], annot_to + filenames[idx])

    print("removed: " + str(i))

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
            
def rename(dataset):
    rootannotfrom ="./dataset/agumented/22agumented/Annotations"
    imagesrootmovefrom = "./dataset/agumented/22agumented/JPEGImages"
    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames

    for idx in range(dataset.__len__()):
        oldfilename = filenames[idx]
        oldimagename = imagenames[idx]

        newfilename = oldfilename[:-4]+"_2.xml"
        newimagename = oldimagename[:-4]+"_2.jpg"
        annopathto = rootannotfrom +"/" +newfilename
        annopathfrom = rootannotfrom + "/"+oldfilename
        os.rename(annopathfrom,annopathto)

        imagepathto = imagesrootmovefrom+"/"+newimagename
        imagepathfrom = imagesrootmovefrom+"/"+oldimagename
        os.rename(imagepathfrom,imagepathto)
        
def killpedestrian(dataset,img_dir,annotations_dir):

    annot_to ='./dataset/final/AnnotationsPed/'
    img_to ='./dataset/final/JPEGImagesPed/'

    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    cls = 27 

    ii = 0
    jj = 0 
    b = True
    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        boxes = targets['boxes']
        print(str(idx) + " out of " +str(dataset.__len__()))


        for idx2, box in enumerate(boxes):
            if targets['labels'][idx2] == cls+1:
                if b:
                   first = idx
                   b = False
                ii +=1
                h = int(box[3]-box[1])
                w = int(box[2]-box[0])        
                if h*w <= 70**2:
                    i = int(box[1])
                    j = int(box[0])
                    h = int(box[3]-box[1])
                    w = int(box[2]-box[0])
                    v = torch.tensor(0)
                    image = T.functional.erase(image,i,j,h,w,v)
                else:
                    ii = 0
                
        tree = ET.parse(os.path.join(annotations_dir,filenames[idx]))
        root = tree.getroot()


        for child in root:
            if child.tag == 'object':
                label = child[0].text
                c4 = child[4]
                width = int(float(c4[2].text))-int(float(c4[0].text))
                height = int(float(c4[3].text))-int(float(c4[1].text))
                testarea = width*height
                if testarea < dataset.area or int(label) not in dataset.takeclass:
                    continue

                if int(label) ==cls:
                    jj +=1
                    if testarea <=70**2:
                        parent = child.getparent()
                        parent.remove(child)
                    else:
                        jj = 0

        save_image(image,img_to + imagenames[idx])
        tree.write(annot_to +filenames[idx])
    

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
        
        
        
        rotator = T.GaussianBlur(9, sigma=(0.1, 2.0))
        image = rotator(org_image)
        save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_blur.jpg")
        tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_blur.xml")

def augmentImages2(dataset, img_dir, annot_dir, img_to, annot_to):

    
    filenames = dataset.annotnames
    imagenames = dataset.imagenames
    
    x = np.array([12,18,22])
    x2 = x + np.ones(len(x))

    x = torch.tensor(x.astype(int))
    x2 = torch.tensor(x2.astype(int))

    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        boxes = targets['boxes']
        print(idx)

        for idx2, box in enumerate(boxes):

            if targets['labels'][idx2] not in x2:
                i = int(box[1])
                j = int(box[0])
                h = int(box[3]-box[1])
                w = int(box[2]-box[0])
                v = torch.tensor(0)
                image = T.functional.erase(image,i,j,h,w,v)
                
        tree = ET.parse(os.path.join(annot_dir,filenames[idx]))
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                label = child[0].text

                if int(label) not in x:
                    parent = child.getparent()
                    parent.remove(child)
        
        
        org_image= image
        # rotator = T.RandomAdjustSharpness(2,1)
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_sharpness.jpg")
        # tree.write(rootannotto +"AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_sharpness.xml")
        
        
        # rotator = T.RandomPerspective(0.2,1)
        # image = rotator(org_image)
        # save_image(image,img_to + "AUGMENTED" +imagenames[idx][:-4]+"_perspective.jpg")
        # tree.write(annot_to + "AUGMENTED" +filenames[idx][:-4]+"_perspective.xml")
        
        
        rotator = T.RandomHorizontalFlip(1)
        image = rotator(org_image)
        save_image(image,img_to +"AUGMENTED" + imagenames[idx][:-4]+"_horizontal.jpg")
        tree.write(annot_to + "AUGMENTED" +filenames[idx][:-4]+"_horizontal.xml")
        
        
        # rotator = T.RandomRotation(15)
        # image = rotator(org_image)
        # save_image(image,imagerootmoveto +"AUGMENTED" + imagenames[idx][:-4]+"_"+str(cls-1)+"_rotation.jpg")
        # tree.write(rootannotto + "AUGMENTED" +filenames[idx][:-4]+"_"+str(cls-1)+"_rotation.xml")
        
        
        
        # rotator = T.GaussianBlur(3, sigma=(0.1, 2.0))
        # image = rotator(org_image)
        # save_image(image,img_to +"AUGMENTED" + imagenames[idx][:-4]+"_blur.jpg")
        # tree.write(annot_to + "AUGMENTED" +filenames[idx][:-4]+"_blur.xml")

def getrounddata(dataset, img_dir, annot_dir, img_to, annot_to):
    imagenames = dataset.imagenames
    filenames = dataset.annotnames
    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])
    x = x + np.ones(len(x))
    resize = T.Resize((299,299))


    for idx in range(dataset.__len__()):
        image, targets = dataset.__getitem__(idx)
        boxes = targets['boxes']
        labels = targets['labels']
        print(str(idx) + " out of " +str(dataset.__len__()))
    
        for idx2, box in enumerate(boxes):
                            
            i = int(box[1])
            j = int(box[0])
            h = int(box[3]-box[1])
            w = int(box[2]-box[0])
            img = T.functional.crop(image,i,j,h,w)
            img = resize(img)

            save_image(img,img_to +str(np.array(labels[idx2])-1)+"_"+ str(idx)+"_"+str(idx2)+".jpg")

        tree = ET.parse(os.path.join(annot_dir,filenames[idx]))
        root = tree.getroot()

        ii = -1
        for child in root:
            ii += 1
            if child.tag == 'object':
                label = child[0].text
                c4 = child[4]
                width = int(float(c4[2].text))-int(float(c4[0].text))
                height = int(float(c4[3].text))-int(float(c4[1].text))
                testarea = width*height
                if testarea < dataset.area or int(label) not in dataset.takeclass:
                    continue

                with open(annot_to +str(label)+"_"+ str(idx)+"_"+str(ii)+".txt", 'w') as f:
                    f.write(str(label))

def cropandsortnonannot(dataset, img_dir, annot_dir, img_to, annot_to):

    imagenames = dataset.imagenames
    filenames = dataset.annotnames
    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])
    x = x + np.ones(len(x))

    for idx in range(dataset.__len__()):
        print(idx)
        image, targets = dataset.__getitem__(idx)        
        
        
        if pd.Series(targets['labels']).isin(x).any() != True:
            print("No correct class")
            print(targets['labels'])
        else:
                
            if len(targets['labels']) != 0:

                image = cropImage(image)        
                save_image(image,img_to + imagenames[idx])
                
                tree = ET.parse(os.path.join(annot_dir,filenames[idx]))
                root = tree.getroot()
                for child in root:
                    if child.tag == 'object':
                        c4 = child[4]
                        c4[1].text = str(int(float(c4[1].text)-800))
                        c4[3].text = str(int(float(c4[3].text)-800))
                tree.write(annot_to + filenames[idx])
            else:
                print("No annotations")

annotations_dir ='./dataset/testset/Annotations/'
img_dir ='./dataset/testset/JPEGImages/'
# annot_to ='./dataset/testset/thirdfoldAnnot/'
# img_to ='./dataset/testset/thirdfoldImages/'
annotations_dir ='./dataset/final/Annotations1600/'
img_dir ='./dataset/final/JPEGImages1600/'

annotations_dir ='./dataset/final/AnnotationsPed/'
img_dir ='./dataset/final/JPEGImagesped/'
annot_to = './dataset/final/inception_V3/AnnotationsRound/'
img_to = './dataset/final/inception_V3/JPEGRound/'


dataset= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area=1600)

getrounddata(dataset,img_dir, annotations_dir, img_to, annot_to)

#augmentImages2(dataset,img_dir, annotations_dir, img_to, annot_to)

#disnonannotimage(dataset, img_dir, annotations_dir, img_to, annot_to)

#sortImages(cls, dataset)
#augmentImages(dataset,cls,rootannotfrom,imagesrootmovefrom)
#rename(dataset)
killpedestrian(dataset, img_dir,annotations_dir)
#disnonannotimage(dataset)
# class nr 


"""
14 0    stop
*2
43 84   no parking         
44 106  no stop or park    
45 138  parking
rotate, random perspective
*4
13 41   give way            
18 38   danger              
rotate, random perspective, horizonflip, sharpness
*10
12 16   priority road       
22 15   uneven road         
rotate, random perspective, horizonflip, sharpness
*1
1  247  30 
17 191  no entry 
38 174  keep right
/3
27 928  pedestrian cross
"""