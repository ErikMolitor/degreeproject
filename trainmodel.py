
from Molitorutils import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

img_dir = './data/totaldataset/JPEGImages/'
annotations_dir = './data/totaldataset/Annotations/'
dataset= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=True),area=700)
dataset_test= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area=700)

split = 0.8

x = [1, 17, 27, 38, 43, 44, 45]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


model, dataset, dataset_test = train(model, dataset, dataset_test, split)



