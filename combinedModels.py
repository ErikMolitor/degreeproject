
import torch.nn as nn
import torchvision.transforms as T 
import torch 
import numpy as np 

def transformOutput(image, pred,thres):

    resize = T.Resize((299,299))
    scores = pred[0]['scores']
    scores_ind = np.argwhere(scores > thres)
    boxes = pred[0]['boxes']
    labels = pred[0]['labels']
    boxes = boxes[scores_ind]
    labels = labels[scores_ind]
    scores = scores[scores_ind]

    imgs = torch.zeros((len(labels),3,299,299))

    for idx, box in enumerate(boxes):
        i = int(box[1])
        j = int(box[0])
        h = int(box[3]-box[1])
        w = int(box[2]-box[0])
        img = T.functional.crop(image,i,j,h,w)
        img = resize(img)
        imgs[idx,:,:,:] = img

    return imgs, labels, scores,boxes

class CombinedModel(nn.Module):
    def __init__(self,modelR, modelS,modelT, modelRetina):
        #super(CombinedModel).__init__()
        self.Rnd = modelR()
        self.Sqr = modelS()
        self.Tri = modelT()
        self.Ret = modelRetina()
    
    def pred(self,image, thres):
        
        output = self.Ret(image)
        images, labels, scores, boxes = transformOutput(image, output, thres)
        
        lb = []

        for idx, image in enumerate(images):
            label = labels[idx]
            if label == 1:
                model = self.Sqr
                x = np.array([1,17, 38, 43, 44])
            elif label == 2:
                model = self.Rnd
                x = np.array([12, 27, 45])
            elif label == 3:
                model = self.Tri
                x = np.array([18,22])
            else:
                cls = 13
                lb.append(cls)
                continue

            out = model(image)
            _, preds = torch.max(out, 1)
            cls = x[preds]
            lb.append(cls)
        
        pred ={
            'boxes':boxes,
            'labels':lb,
            'scores':scores
        }

        return pred
            

            
        


