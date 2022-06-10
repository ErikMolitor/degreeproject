
import torchvision
from Molitorutils import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import numpy as np


if __name__ == '__main__':
    
    img_dir = './dataset/croped/JPEGImages/'
    annotations_dir = './dataset/croped/Annotations/'

    img_dir = './dataset/dataForTesting/Telestaden/PanoRensad/'
    img_dir = './dataset/german/JPEGImages/'
    img_dir = './dataset/presentation/'

    annotations_dir = './dataset/croped/Annotations/'
    
    # annotations_dir ='./dataset/final/AnnotationsPed/'
    # img_dir ='./dataset/final/JPEGImagesPed/'

    
    # annotations_dir ='./dataset/testdata2/Annotations/'
    # img_dir ='./dataset/testdata2/JPEGImages/'
    

    x = np.array([1,12,13,17,18,22, 27, 38, 43, 44, 45])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset= pascalVoc(img_dir, annotations_dir,takeclass=x, transform=get_transform(train=True))
    num_classes = len(dataset.takeclass) +1

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load('./savedmodels/Faster/modelparams_504.pt'))
    model.eval()

    # define training and validation data loaders
    #data_loader = torch.utils.data.DataLoader(
    #    dataset, batch_size=1, shuffle=True, num_workers=2,
    #    collate_fn=utils.collate_fn)
    
    threshold = 0.2

    pred, scores  = displayone(model,1 , dataset, threshold,x)




"""
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
 """






