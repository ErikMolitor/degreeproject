
from Molitorutils import *
import matplotlib.pyplot as plt


annotations_dir = './dataset/final/AnnotationsPed/' 
img_dir = './dataset/final/JPEGImagesPed/'

dataset= pascalVoc(img_dir, annotations_dir, transform=get_transform(train=False),area=700)
#600 good

image, targets = dataset.__getitem__(603)



plt.rcParams['figure.figsize'] = [10, 5]

plotBB(image, targets,width=15)


