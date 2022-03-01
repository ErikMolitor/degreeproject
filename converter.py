#!/usr/bin/env python
# coding:utf-8
 
from ast import Break
from operator import truediv
from tkinter.tix import Tree
from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import numpy as np
from PIL import Image
import os 
import xml.etree.ElementTree as ET



def xmlconverter():
        
    root = "./data/german dataset/FullIJCNN2013/"

    with open(root + "gt.txt","r") as f:
        lines = f.readlines() 
        i = 0
        
        while i <= len(lines)-1:
            
            lst = []
            lst.append(lines[i].split(';'))
            ii = i
            
            
            picName = lines[i].split(';')[0]

            if i != len(lines)-1:
                nextName = lines[i+1].split(';')[0]
                while picName == nextName:
                    i +=1
                    lst.append(lines[i].split(';'))   
                    picName = lines[i].split(';')[0]
                    nextName = lines[i+1].split(';')[0]
                    
            
            
            node_root = Element('annotation')

            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'TrainIJCNN2013'
            
            # HERE       
            line=lines[i].split(';')
            line_shape = np.reshape(line[1:],(-1,5))



            node_filename = SubElement(node_root, 'filename')
            img_name = lst[0][0] 
            node_filename.text = img_name

            node_source = SubElement(node_root, 'source')

            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = '1360'

            node_height = SubElement(node_size, 'height')
            node_height.text = '800'

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = '3'

            node_segment = SubElement(node_root,'segmented')
            node_segment.text = '0'


            for j in range(len(lst)):
                
                node_object = SubElement(node_root, 'object')

                node_name = SubElement(node_object, 'name')
                class_n=lst[j][5][:-1]
                node_name.text = str(class_n)
                
                node_trunc = SubElement(node_object, 'truncated')
                node_trunc.text = "0"
                node_occl = SubElement(node_object, 'occluded')
                node_occl.text = "0"
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = lst[j][1]
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = lst[j][2]
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = lst[j][3]
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = lst[j][4]
            Xml = tostring(node_root, pretty_print=True)  #Formatted display, the newline of the newline
            #dom = parseString(Xml)

            with open("data/german dataset/PascalVoc/Annotations/"+img_name[:-4]+".xml","wb") as f:
                f.write(Xml)
            i+=1 

def ppm2jpg():
    root ="./data/german dataset/FullIJCNN2013/"
    root2 = "./data/german dataset/PascalVoc/JPEGImages/"
    filenames = sorted([f for f in os.listdir(root) if f.endswith('.ppm')])    

    for i in range(len(filenames)):
        im = Image.open(root +filenames[i])
        print(filenames[i][:-4]+".jpg")

        im.save(root2+filenames[i][:-4]+".jpg")       
                
                

def xmledit():
    root = "./data/german dataset/PascalVoc/editedAnnotations"
    annotnames = sorted(os.listdir(root))
    
    
    for idx in range(1):
        file = ET.parse(os.path.join(root,annotnames[0])).getroot()
        node_source = ET.SubElement(file, 'source')
        tree = ET.ElementTree(file)
        print(file[0])

        tree.write("./data/german dataset/PascalVoc/test/"+annotnames[idx])

        

xmlconverter()
    

                
                
                
                
                
                