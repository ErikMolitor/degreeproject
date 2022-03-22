
from Molitorutils import *
import torch
from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
import pickle
import time

def main(dataset,dataset_test, model,split):
    model.train()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    # use our dataset and defined transformations
    
    torch.manual_seed(1)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = round(len(indices)*split)
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

    batch_size = 2

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn, 
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
       dataset_test, batch_size=batch_size, 
       shuffle=False, num_workers=2,
       collate_fn=utils.collate_fn,pin_memory=True)
    
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   #step_size=3,
                                                   #gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for 10 epochs
    losses = []
    classification = []
    bbox_regression = []

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

    num_epochs = 200

    for epoch in range(0,num_epochs):
        start = time.time()
        # train for one epoch, printing every 10 iterations
        metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)  
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        losses.append(float(str(metrics.meters['loss']).split(" ")[0]))
        classification.append(float(str(metrics.meters['classification']).split(" ")[0]))
        bbox_regression.append(float(str(metrics.meters['bbox_regression']).split(" ")[0]))

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

        if epoch %5 == 4: # save every 5th epoch model
            torch.save(model.state_dict(), './savedmodels/RetinaNet/modelparams_'+str(epoch)+'.pt')

        with open('./savedStats/RetinaNet/stats_'+str(epoch)+'.pickle', 'wb') as f:
           pickle.dump([losses, classification, bbox_regression, stat0, stat1, stat2, stat3,
                           stat4, stat5, stat6, stat7, stat8, stat9, stat10, stat11], f)
        end =time.time()
        dt = (end-start)
        dtend = dt*(num_epochs-(epoch+1))
        print('Epoch complete in {:.0f}m {:.0f}s'.format(dt // 60, dt % 60))
        print('Estimated complete in {:.0f} days {:.0f}h {:.0f}m {:.0f}s'.format(dtend//60//60//24, dtend // 60//60%24,dtend // 60%60, dtend % 60))
    
    return model, dataset, dataset_test



if __name__ == '__main__':

    annotations_dir ='./dataset/final/AnnotationsPed/'
    img_dir ='./dataset/final/JPEGImagesPed/'

    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45]) # just add 14 stop

    dataset= pascalVocRetina(img_dir, annotations_dir,takeclass=x, transform=get_transform(train=True))
    dataset_test = pascalVocRetina(img_dir, annotations_dir,takeclass=x,transform=get_transform(train=False))
    num_classes = int(5) # 4+1

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone = True,num_classes=num_classes )
    

    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    #model.roi_heads.box_predictor = RetinaNet(in_features, num_classes)
    

    #model.load_state_dict(torch.load('./savedmodels/modelparams_2.pt'))

    split  = 0.8
    model, dataset, dataset_test = main(dataset,dataset_test, model,split)
    torch.save(model.state_dict(), 'modelparamslast.pt')


