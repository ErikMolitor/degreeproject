from Molitorutils import *
import sys 
sys.path.append('C:\\Users\\SEROEM\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages')
from detection.engine import evaluate

def evaluateerik(model, data_loader, device,thres):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    
    for batch, (images, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            pred = model(images)
        pred[0]
        scores = pred[0]['scores'].to('cpu')
        scores_ind = np.argwhere(scores > thres)
        pred_boxes = pred[0]['boxes']
        pred_boxes = pred_boxes[scores_ind]
        print("NEW")
        print(pred_boxes)
        print(targets[0]['boxes'])
        mse = torch.nn.MSELoss()
        loss = mse(pred_boxes,targets[0]['boxes'])
        print(loss)


    # gather the stats from all processes
    torch.set_num_threads(n_threads)
    return 


    
def eval_train(model, dataset,dataset_test, split,thres):
    model.train()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = dataset.takeclass.size+1

    # split the dataset in train and test set
    torch.manual_seed(1)

    indices = torch.randperm(len(dataset)).tolist()
    num_train = round(len(indices)*split)
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, 
        shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn, 
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, 
        shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn,
        pin_memory=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    #evaluateerik(model,data_loader_test,device,thres)
    evaluate(model,data_loader_test,device)
    """     num_epoch = 10
    start_time = time.time()
    for epoch in range(num_epoch):
        model_time = time.time()
        train_1_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()

        time_diff = time.time() - model_time
        print("TIME FOR EPOCH: " + str(epoch)+ " is: " + str(time_diff/60))

    print("Total time for traning: " + str((time.time()-start_time)/60)) """

    return model, dataset, dataset_test

    

if __name__ == '__main__':

    import pycocotools

    print(sys.path)
    img_dir = './dataset/croped/JPEGImages/'
    annotations_dir = './dataset/croped/Annotations/'

    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset= pascalVoc(img_dir, annotations_dir,takeclass=x, transform=get_transform(train=True))
    dataset_test = pascalVoc(img_dir, annotations_dir,takeclass=x,transform=get_transform(train=False))

    num_classes = len(dataset.takeclass) +1


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load('modelparamsBias27.pt'))
    model.eval()


    thres = 0.2
    eval_train(model,dataset,dataset,split = 0.8,thres=0.1)

    #pred, scores  = displayone(model, 400, dataset, threshold)

    #print(pred)