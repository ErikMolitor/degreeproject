from Molitorutils import *

import copy
import pickle
import torchvision.transforms as T
import random

class inceptiondata(Dataset):
   def __init__(self, img_dir, annotations_dir,train, area = 400 ):
        self.annotation_dir = annotations_dir
        self.img_dir = img_dir
        self.transform = torchvision.transforms.ToTensor()
        self.train = train
        self.imagenames = sorted(os.listdir(self.img_dir))
        self.annotnames = sorted(os.listdir(self.annotation_dir))
        self.length = len(os.listdir(self.annotation_dir))
        self.randomcrop = T.RandomCrop(size=(128, 128))
        self.resize = T.Resize((299,299))
        self.rotate = T.RandomRotation(15)
        self.autoaug = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
   def __len__(self):
       return self.length
 
   def __getitem__(self, idx):
        image = Image.open(self.img_dir +self.imagenames[idx]).convert("RGB")
        if self.train:
            image = self.autoaug(image)

        with open(self.annotation_dir+self.annotnames[idx]) as f:
           label = int(f.readlines()[0])
        image = self.transform(image)
        label = torch.tensor(label)
        image = image[0]


        return image, label


def train_model(model, dataloaders, criterion, optimizer,x,cls, num_epochs=25, is_inception=False):
    since = time.time()
    num_labels = len(x)
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_val = []
    epoch_acc_val = []

    with open('./savedStats/'+cls+'2/stats_'+cls+'_'+str(999)+'.pickle', 'rb') as f:
        epoch_loss_train, epoch_acc_train,epoch_loss_val,epoch_acc_val = pickle.load(f)

    for epoch in range(1000,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                images = list(image.to(device) for image in inputs)
                img = torch.zeros((len(images),3,299,299)).to(device)
                for idx1, image in enumerate(images):
                    img[idx1,:,:,:] = image.to(device)
                inputs = img

                labels = labels.to(device)
                y = np.zeros((len(labels),num_labels))
                for idx,label in enumerate(labels):
                    ind = list(x).index(label)
                    y[idx,ind] = 1
                labels2 = torch.from_numpy(y).to(device)
                labels2 = labels2.float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels2)
                        loss2 = criterion(aux_outputs, labels2)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels2)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                for idx, lb in enumerate(preds):
                    if x[lb]==labels[idx]:
                        running_corrects +=torch.tensor(1)
               
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                epoch_loss_train.append(epoch_loss)
                epoch_acc_train.append(epoch_acc)
            if phase == 'val':                
                epoch_loss_val.append(epoch_loss)
                epoch_acc_val.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        if epoch %20 == 19 or epoch == 0:
            with open('./savedStats/'+str(cls)+'2/stats_'+str(cls)+'_'+str(epoch)+'.pickle', 'wb') as f:
                pickle.dump([epoch_loss_train, epoch_acc_train,epoch_loss_val,epoch_acc_val], f)
            
            torch.save(model.state_dict(), './savedmodels/'+str(cls)+'2/modelparams_'+str(cls)+'_'+str(epoch)+'.pt')
        
        end = time.time()
        dt = end -start

        print()
        dtt = (end-since)/(epoch+1)
        dtend = dtt*(num_epochs-(epoch+1))
        
        dtend = dt*(num_epochs-(epoch+1))

        print('Best val Acc: {:4f}'.format(best_acc))
        print('Epoch complete in {:.0f}m {:.0f}s'.format(dt // 60, dt % 60))
        print('Estimated complete in {:.0f} days {:.0f}h {:.0f}m {:.0f}s'.format(dtend//60//60//24, dtend // 60//60%24,dtend // 60%60, dtend % 60))
        print()
        print()


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



if __name__ == '__main__':
    
    #cls = "Round"
    #cls = "Square"
    cls = "Triangular"

    if cls == "Round":
        #round2
        x = np.array([1,17, 38, 43, 44]) # just add 14 stop
    elif cls == "Square":
        #square
        x = np.array([12, 27, 45]) # just add 14 stop
    elif cls == "Triangular":
        #triangular
        x = np.array([18,22]) # just add 14 stop

    annotations_dir ='./dataset/final/inception_V3/Annotations'+str(cls)+'/'
    img_dir ='./dataset/final/inception_V3/JPEGImages'+str(cls)+'/'

    dataset= inceptiondata(img_dir, annotations_dir,train=True)
    dataset_test = inceptiondata(img_dir, annotations_dir,train=False)
    
    #Model set up
    model = torchvision.models.inception_v3(pretrained=False,aux_logits=True)
    ### ResNet or Inception
    classifier_input = model.fc.in_features
    num_labels = len(x)
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_labels)
    # Replace default classifier with new classifier
    model.fc = torch.nn.Linear(classifier_input, num_labels)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.load_state_dict(torch.load('./savedmodels/'+str(cls)+'2/modelparams_'+str(cls)+'_999.pt'))
    
    split  = 0.8
    torch.manual_seed(1)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = round(len(indices)*split)
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

    batch_size = 4

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2,
        pin_memory=True,drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
       dataset_test, batch_size=batch_size, 
       shuffle=False, num_workers=2,
       pin_memory=True,drop_last=True)

    dataloaders = {
        'train': data_loader,
        'val':data_loader_test
    }
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001,
                                momentum=0.9)
    criterion = torch.nn.MSELoss(0.001)
    num_epochs = 3000
    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, x,cls,num_epochs, is_inception=True)

