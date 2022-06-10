
from Molitorutils import *
import torchvision
import torch 
import numpy as np 
from detection.engine import evaluate
import pickle


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
    
    leng = len(os.listdir(dir))
    
    for i in range(leng):
        print('--------------')
        print('Epoch: '+str(i)+'/'+str(leng))
        start = time.time()
        nr = 4 +i*5
        
        model.load_state_dict(torch.load('./savedmodels/Faster/modelparams_'+str(nr)+'.pt',map_location = torch.device('cpu')))
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
        
        if nr >= 900:
            with open('./savedStats/MixedDataFaster2/stats_'+str(nr)+'.pickle', 'wb') as f:
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



if __name__ == '__main__':
    
    annotations_dir ='./dataset/testdata2/Annotations/'
    img_dir ='./dataset/testdata2/JPEGImages/'
    
    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])
    
    
    device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    dataset= pascalVoc(img_dir, annotations_dir, takeclass = x, transform=get_transform(train=False),area=1600)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = dataset.takeclass.size +1

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    dir = './savedmodels/Faster/'

    batch_size = 1
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn, 
        pin_memory=True)
    testing(model,data_loader,dir)