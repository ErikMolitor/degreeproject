
from matplotlib import image
from Molitorutils import *
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import detection.utils as utils
import pickle
import time

class inceptiondata(Dataset):
   def __init__(self, img_dir, annotations_dir, area = 400, target_transform = None ):
       self.annotation_dir = annotations_dir
       self.img_dir = img_dir
       self.target_transform = target_transform
       self.transform = T.ToTensor()

       self.imagenames = sorted(os.listdir(self.img_dir))
       self.annotnames = sorted(os.listdir(self.annotation_dir))
       self.length = len(os.listdir(self.annotation_dir))
      
   def __len__(self):
       return self.length
 
   def __getitem__(self, idx):
        image = Image.open(self.img_dir +self.imagenames[idx]).convert("RGB")
        with open(self.annotation_dir+self.annotnames[idx]) as f:
           label = int(f.readlines()[0]) 

        image = self.transform(image)
        label = torch.tensor(label)
        image = image[0]

        return image, label

def train_1_epoch(model,data_loader,device,optimizer,criterion,epoch):
    model.train()

    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]

        images = list(image.to(device) for image in images)
        img = torch.zeros((len(images),3,299,299)).to(device)
        for idx1, image in enumerate(images):
            img[idx1,:,:,:] = image.to(device)
        
        images = img
        
        y = np.zeros((len(labels),num_labels))
        for idx,label in enumerate(labels):
            ind = list(x).index(label)
            y[idx,ind] = 1
        labels = y
        labels = torch.from_numpy(labels).to(device)
        labels.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)

        labels = labels.float()

        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        printFrq = round(len(data_loader)/4)
        if i % printFrq == printFrq-1:    # print every 10 mini-batches
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1:5d}/{len(data_loader)}] loss: {running_loss / printFrq:.6f}')
            losslastepoch = running_loss
            running_loss = 0.0
    return losslastepoch

def evaluateOwn(model, dataloader_test,device):
    model.eval()

    correct = 0
    total = 0

    for images, labels in dataloader_test:
        
        images = images.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        print("PRED")
        print(predicted)
        print("")
        print(torch.max(outputs.data, 1))
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
        
        
    print('Accuracy of test images: %f %%' % (100 * float(correct) / total))

    return 100 * float(correct) / total

def main(dataset,dataset_test, model,split,x):
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    #device = torch.device('cpu')

    # our dataset has two classes only - background and person
    # use our dataset and defined transformations
    
    torch.manual_seed(1)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = round(len(indices)*split)
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])


    batch_size = 8

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
       dataset_test, batch_size=batch_size, 
       shuffle=False, num_workers=2,
       pin_memory=True)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001,
                                momentum=0.9)
    criterion = torch.nn.MSELoss(0.001)

    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   #step_size=3,
                                                   #gamma=0.5)


    losslast = []
    testvalue = []
    num_epochs = 1000


    i = 0
    for epoch in range(0,num_epochs):
        start = time.time()

       # metric = train_1_epoch(model,data_loader,device,optimizer,criterion,epoch)
        
        perc = evaluateOwn(model, data_loader_test,device) 
        
        losslast.append(0)
        testvalue.append(perc)

        if i %5 == 4:
            torch.save(model.state_dict(), './savedmodelRound/modelparams_round_'+str(epoch)+'.pt')

        with open('./savedStatsRound/stats_round_'+str(epoch)+'.pickle', 'wb') as f:
            pickle.dump([losslast, testvalue], f)
    
        end =time.time()
        timediff = (end-start)/60
        print("Time for epoch: " + str(timediff))
    
    return model, dataset, dataset_test

if __name__ == '__main__':

    cls = "Round"
    #cls = "Square"
    #cls = "Triangular"

    if cls == "Round":
        #round
        x = np.array([1,17, 38, 43, 44]) # just add 14 stop
    elif cls == "Square":
        #square
        x = np.array([12, 27, 45]) # just add 14 stop
    elif cls == "Triangular":
        #triangular
        x = np.array([13,18,22]) # just add 14 stop

    annotations_dir ='./dataset/final/inception_V3/Annotations'+str(cls)+'/'
    img_dir ='./dataset/final/inception_V3/JPEGImages'+str(cls)+'/'

    

    dataset= inceptiondata(img_dir, annotations_dir)
    dataset_test = inceptiondata(img_dir, annotations_dir)
    
    #Model set up
    model = torchvision.models.inception_v3(pretrained=False,aux_logits=False)
    ### ResNet or Inception
    classifier_input = model.fc.in_features
    num_labels = len(x)
    # Replace default classifier with new classifier
    model.fc = torch.nn.Linear(classifier_input, num_labels)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #model.load_state_dict(torch.load('./savedmodels/modelparams_2.pt'))
    
    split  = 0.8
    model, dataset, dataset_test = main(dataset,dataset_test, model,split,x)
    torch.save(model.state_dict(), './savedmodelRound/modelparamslast.pt')


