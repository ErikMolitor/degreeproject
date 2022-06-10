
from Molitorutils import *
import torchvision
import torch 
import numpy as np 
from detection.engine import evaluate
import pickle
import torchvision.transforms as T 
from tkinter import *
from torchvision.utils import save_image
import random

def testing(model, data_loader_test,dir):
    
    losses = []
    loss_box_reg = []
    loss_rpn_box_reg = []
    loss_classifier = []
    loss_objectness = []
    stat0 = []
    stat1 = []
    stat2 = []
    stat3 = []
    stat4 = []
    stat5 = []
    stat6 = []
    stat7 = []
    stat8 = []
    stat9 = []
    stat10 = []
    stat11 = []
    dir = "./savedmodels/RetinaNet/"

    leng = len(os.listdir(dir))
    
    for i in range(leng):
        print('--------------')
        print('Epoch: '+str(i)+'/'+str(leng))
        start = time.time()
        nr = 4 +i*5
        model.updateRet(nr)
       
        cocoeval = evaluate(model, data_loader_test, device=device)
        #Stat object is from pycocotools' self.stats in summarize()
        #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        stat = cocoeval.coco_eval['bbox'].stats
        
        #Append all stats
        stat0.append(stat[0])
        stat1.append(stat[1])
        stat2.append(stat[2])
        stat3.append(stat[3])
        stat4.append(stat[4])
        stat5.append(stat[5])
        stat6.append(stat[6])
        stat7.append(stat[7])
        stat8.append(stat[8])
        stat9.append(stat[9])
        stat10.append(stat[10])
        stat11.append(stat[11])
        
        with open('./savedStats/MixedDataCombined4/stats_'+str(nr)+'.pickle', 'wb') as f:
           pickle.dump([losses, loss_box_reg, loss_rpn_box_reg, loss_classifier, loss_objectness, stat0, stat1, stat2, stat3,
                           stat4, stat5, stat6, stat7, stat8, stat9, stat10, stat11], f)
           
        end =time.time()
        dt = (end-start)
        print()
        dtend = dt*(leng-(i+1))

        print('Epoch complete in {:.0f}m {:.0f}s'.format(dt // 60, dt % 60))
        print('Estimated complete in {:.0f} days {:.0f}h {:.0f}m {:.0f}s'.format(dtend//60//60//24, dtend // 60//60%24,dtend // 60%60, dtend % 60))
        print()
        print()

class MyModel(torch.nn.Module):
    def __init__(self,modelR, modelS,modelT, modelRetina):
        super(MyModel,self).__init__()
        self.Rnd = modelR
        self.Sqr = modelS
        self.Tri = modelT
        self.Ret = modelRetina

    def forward(self,image):
        thres = 0.5
        output = self.Ret(image)
        images, labels, scores, boxes = self.transformOutput(image, output, thres)
        lb = []
        labels = labels[0]
        FullX = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])

        for idx, image in enumerate(images):
            label = labels[idx]

            if label == 1:
                model = self.Sqr
                x = np.array([12, 27, 45]) 
            elif label == 2:
                model = self.Rnd
                x = np.array([1,17, 38, 43, 44])
            elif label == 3:
                model = self.Tri
                x = np.array([18,22])
            else:
                cls = 13
                lb.append(cls)
                continue
            
            image = image.unsqueeze(0)
            out = model(image)
            _, preds = torch.max(out, 1)
            cls = x[preds]

            # s = scores[0][idx]
            # s = s.cpu().detach().numpy()
            #save_image(image,'./test/' +"Cut_"+ str(cls)+"_"+str(s)+"_"+str(random.randint(0,1000))+".jpg")

            cls = list(FullX).index(int(cls))+1 

            lb.append(cls)

        pred = {}
        pred["boxes"] = boxes[0]
        pred["scores"] = scores[0]
        pred["labels"] = torch.IntTensor(lb)
        
        pred = [pred]

        return pred
    def updateRet(self,nr):
        self.Ret.load_state_dict(torch.load('./savedmodels/RetinaNet/modelparams_'+str(nr)+'.pt',map_location = torch.device('cpu')))
        

    def transformOutput(self, image, pred,thres):
        
        resize = T.Resize((299,299))
        scores = pred[0]['scores']
        scores_ind = np.argwhere(scores.cpu() > thres)
        boxes = pred[0]['boxes']
        labels = pred[0]['labels']
        boxes = boxes[scores_ind]
        labels = labels[scores_ind]
        scores = scores[scores_ind]
    
        imgs = torch.zeros((len(labels[0]),3,299,299))
        image = image[0]

        for idx in range(len(boxes[0])):
            box = boxes[0][idx]
            i = int(box[1])
            j = int(box[0])
            h = int(box[3]-box[1])
            w = int(box[2]-box[0])
            img = T.functional.crop(image,i,j,h,w)
            img = resize(img)
            imgs[idx,:,:,:] = img
        
        imgs = imgs.to(device)

        return imgs, labels, scores,boxes
           

if __name__ == '__main__':
    
    annotations_dir ='./dataset/testdata2/Annotations/'
    img_dir ='./dataset/testdata2/JPEGImages/'

    # annotations_dir ='./dataset/final/AnnotationsPed/'
    # img_dir ='./dataset/final/JPEGImagesPed/'

    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')


    dataset= pascalVoc(img_dir, annotations_dir, takeclass = x, transform=get_transform(train=False),area=1600)
    num_classes = int(5) # 4+1
    incepNr = 2999
    modelRet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone = True,num_classes=num_classes )
    modelRet.load_state_dict(torch.load('./savedmodels/RetinaNet/modelparams_4.pt'))
    modelRet.to(device)

    modelRound = torchvision.models.inception_v3(pretrained=False,aux_logits=True)
    classifier_input = modelRound.fc.in_features
    num_labels = int(5)
    num_ftrs = modelRound.AuxLogits.fc.in_features
    modelRound.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_labels)
    modelRound.fc = torch.nn.Linear(classifier_input, num_labels)
    modelRound.load_state_dict(torch.load('./savedmodels/Round2/modelparams_round_'+str(incepNr)+'.pt'))
    modelRound.to(device)

    modelSquare = torchvision.models.inception_v3(pretrained=False,aux_logits=True)
    classifier_input = modelSquare.fc.in_features
    num_labels = int(3)
    num_ftrs = modelSquare.AuxLogits.fc.in_features
    modelSquare.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_labels)
    modelSquare.fc = torch.nn.Linear(classifier_input, num_labels)
    modelSquare.load_state_dict(torch.load('./savedmodels/Square2/modelparams_Square_'+str(incepNr)+'.pt'))
    modelSquare.to(device)

    modelTriangular = torchvision.models.inception_v3(pretrained=False,aux_logits=True)
    classifier_input = modelTriangular.fc.in_features
    num_labels = int(2)
    num_ftrs = modelTriangular.AuxLogits.fc.in_features
    modelTriangular.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_labels)
    modelTriangular.fc = torch.nn.Linear(classifier_input, num_labels)
    modelTriangular.load_state_dict(torch.load('./savedmodels/Triangular2/modelparams_Triangular_'+str(incepNr)+'.pt'))
    modelTriangular.to(device)
    
    modelRound.eval()
    modelRet.eval()
    modelSquare.eval()
    modelTriangular.eval()

    batch_size = 1
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn, 
        pin_memory=True)


    model = MyModel(modelRound,modelSquare,modelTriangular,modelRet)
    #model = modelRet
    thres = 0.5
    
    #image,target = next(iter(data_loader))

    # pred = model(image)
    # print(pred)


    #pred, scores = displayone(model,2,dataset,thres,x)

    # for image, targets in data_loader:
    #     image = list(img.to(device) for img in image)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     torch.cuda.synchronize()
    #     model_time = time.time()
    #     outputs = model(image)
    #     print()
    #     print("NEW")
    #     print(targets[0]['labels'])
    #     print(outputs[0]['labels'])

    testing(model,data_loader,device)
    # for idx in range(100):
    #     print('index: '+ str(idx))

    #     image, targets = dataset.__getitem__(idx)
    #     image = list([image])
    #     #model = model.to('cpu')
    #     pred = model(image)
    #     threshold = 0.5
        
    #pred, scores  = displayone(model, 53, dataset, threshold,x)
