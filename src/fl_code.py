


import os
import pandas as pd
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.models
import cv2
import copy
import argparse

from torchvision.transforms import Lambda, Normalize 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from torchvision import models
from torchvision.models import resnet18
from monai.transforms import (
    Activations,
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity, Transpose, 
    LoadImage,
    ToTensor,
    Resize,
)

'''
args=argparse.ArgumentParser()
args.add_argument('num_users',help='number of users')
args.add_argument('Train file names',nargs="+")
args.add_argument('Test file names',nargs='+')
args.add_argument('Val file names',nargs="+")
'''
 








'''
path_files_colu_train=np.concatenate((path_files_colu_train,path_files_rcol_train),axis=0)
path_files_corn_train=np.concatenate((path_files_corn_train,path_files_rcor_train),axis=0)

labels_colu_train=np.concatenate((labels_colu_train,labels_rcol_train),axis=0)
labels_corn_train=np.concatenate((labels_corn_train,labels_rcor_train),axis=0)
'''

#Provide your own list of path and labels for different institutions
#path_file_corn_train contains paths to train images for insti CORN
pathFileTrain= [path_files_corn_train ,path_files_oshu_train ,path_files_beau_train , path_files_ceda_train , path_files_chila_train , path_files_colu_train , path_files_miam_train]
labelTrain= [labels_corn_train , labels_oshu_train , labels_beau_train , labels_ceda_train , labels_chila_train ,labels_colu_train ,labels_miam_train ]

'''
path_files_colu_test=np.concatenate((path_files_colu_test,path_files_rcol_test),axis=0)
path_files_corn_test=np.concatenate((path_files_corn_test,path_files_rcor_test),axis=0)

labels_colu_test=np.concatenate((labels_colu_test,labels_rcol_test),axis=0)
labels_corn_test=np.concatenate((labels_corn_test,labels_rcor_test),axis=0)
'''

pathFileTest= [path_files_corn_test , path_files_oshu_test , path_files_beau_test , path_files_ceda_test , path_files_chila_test , path_files_colu_test , path_files_miam_test ]
labelTest= [labels_corn_test , labels_oshu_test , labels_beau_test , labels_ceda_test , labels_chila_test ,labels_colu_test ,labels_miam_test ]

'''
path_files_colu_val=np.concatenate((path_files_colu_val,path_files_rcol_val),axis=0)
path_files_corn_val=np.concatenate((path_files_corn_val,path_files_rcor_val),axis=0)
labels_colu_val=np.concatenate((labels_colu_val,labels_rcol_val),axis=0)
labels_corn_val=np.concatenate((labels_corn_val,labels_rcor_val),axis=0)
'''

pathFileValid= [path_files_corn_val , path_files_oshu_val , path_files_beau_val , path_files_ceda_val , path_files_chila_val , path_files_colu_val , path_files_miam_val ]
labelValid= [labels_corn_val , labels_oshu_val , labels_beau_val , labels_ceda_val , labels_chila_val ,labels_colu_val ,labels_miam_val  ]


transforms = Compose(
    [
        LoadImage(image_only=True),
        # Resize((480,640)),
        AddChannel(), 
        ScaleIntensity(),
        ToTensor(),
        Lambda(lambda x: torch.cat([x, x, x], 0)),
        # ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
)
act = Activations(softmax=True)

class RaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index], self.image_files[index]




class NetArch(nn.Module):
	def __init__(self):
		super(NetArch, self).__init__()
		self.netarch = torchvision.models.resnet18(pretrained=True)
		num_ftrs = self.netarch.fc.in_features
		self.netarch.fc = nn.Sequential(
			nn.Linear(num_ftrs, 3))

	def forward(self, x):
		x = self.netarch(x)
		return x

#federated Averaging 
def FedAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg

def train(dataloader,model,insti_name,datasetTrain,datasetValid,dataLoaderValid,trMaxEpoch=1):
    '''
    Takes the a copy of the current global mode and trains on the data of a insitution
    for one epoch; Dataloaders in the arguments are dataloaders of the instituion you want
    to train on 
    '''
    learning_rate=1e-4
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epochID in range(0, trMaxEpoch):
        batchcount=0
        total_correct=0
        print("Training model with data of institution "+ insti_name)
        print("Epoch " + str(epochID), end =" ")
        running_loss_train = 0.0
        num_elements=0
        for batch in dataloader:
        #print(batchcount, end=" ") 
            model.train()
            inputs = batch[0].to(device)
            num_elements+=inputs.shape[0]
            labels = batch[1].to(device)
            optimizer.zero_grad() 
            with torch.set_grad_enabled(True): # enable gradient while training
                outputs = model(inputs)
                trainloss = criterion(outputs, labels)
                trainloss.backward()
                optimizer.step()
            #lossFile.write("Batch " + str(batchcount) + " train loss = " + str(trainloss.item()) + "\n")
            total_correct += torch.eq(torch.argmax(act(outputs),dim=1), labels).sum().item()
            batchcount+=1
            #lossFile.flush()
            running_loss_train += trainloss.item()*inputs.size(0)
        acc_metric_tr=float(total_correct)/num_elements
    # Check validation loss after each epoch
        model.eval() # Evaluation mode
        running_loss_val = 0.0
        batchcount=0
        num_elements=0
        total_corr=0
        for batch in dataLoaderValid:
        #print(batchcount, end=" ")
            batchcount+=1
            inputs = batch[0].to(device)
            num_elements+=inputs.shape[0]

            labels = batch[1].to(device)
            with torch.set_grad_enabled(False): # don't change the gradient for validation
                outputs = model(inputs)
                validloss = criterion(outputs, labels)
            running_loss_val += validloss.item()*inputs.size(0)	
            total_corr+= torch.eq(torch.argmax(act(outputs),dim=1), labels).sum().item()
        acc_metric_val=float(total_corr)/num_elements
        epoch_loss_train = running_loss_train / len(datasetTrain)
        epoch_loss_val = running_loss_val / len(datasetValid)

        print('Train Loss: {:.4f} Val Loss: {:.4f}'.format(epoch_loss_train, epoch_loss_val))
        print('Train Accuracy:{:.4f} Val Accuracy:{:.4f}'.format(acc_metric_tr,acc_metric_val))
    return model.state_dict()

# set up parameters
device = torch.device(0 if torch.cuda.is_available() else "cpu")
num_epochs=10
num_users=7
learning_rate = 1e-4
global_model=NetArch().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(global_model.parameters(), learning_rate)
batch_size=64

#Set up dataloader for different instituions 
all_datasets_train=[]
all_datasets_test=[]
all_datasets_valid=[]

all_dataloaders_test=[]
all_dataloaders_valid=[]
all_dataloaders_train=[]

for i in range(7): #7 is the num of users in our case 
#PathFileTrain is a list of lists; pathFile[0] contains list of path_files to data in instition 0
    datasetTrain=RaceDataset(pathFileTrain[i],labelTrain[i],transforms)
    all_datasets_train.append(datasetTrain)
    datasetTest=RaceDataset(pathFileTest[i],labelTest[i],transforms)
    datasetValid=RaceDataset(pathFileValid[i],labelValid[i],transforms)
    all_datasets_valid.append(datasetValid)
    all_datasets_test.append(datasetTest)
    all_dataloaders_train.append(DataLoader(dataset=datasetTrain, batch_size=batch_size, shuffle=False, num_workers=4))
    all_dataloaders_test.append(DataLoader(dataset=datasetTest, batch_size=batch_size, num_workers=4))
    all_dataloaders_valid.append(DataLoader(dataset=datasetValid, batch_size=batch_size, num_workers=4))

#Have a global validation set
global_pathFileValid=[]
global_labels_valid=[]

insti_map={0:'CORN', 1:'OSHU', 2:'BEAU', 3:'CEDA', 4:'CHILA', 5:'COLU', 6:'MIAM'}

for i in range(len(pathFileValid)):
  global_pathFileValid.extend(pathFileValid[i])
  global_labels_valid.extend(labelValid[i])
datasetValid=RaceDataset(global_pathFileValid,global_labels_valid,transforms)
dataloader_global_val=DataLoader(dataset=datasetValid, batch_size=batch_size, num_workers=4)

#Run Federated round till gloabl validation decreases
num_Federated_round=50 # adjust this or monitor val loss to stop
bestvalloss = 100
modelname="Final_Federated_model"
for ep in range(num_Federated_round):
    print("Running Federated round: ",ep+1)
    # train the global model for each instituation seperately and average the weights
    weights=[]
    for usr in range(num_users):
        #train function trains the model on the data of a particular institution and returns the state dict
        w=train(all_dataloaders_train[usr],copy.deepcopy(global_model),insti_map[usr],all_datasets_train[usr],all_datasets_valid[usr],all_dataloaders_valid[usr])
        weights.append(w)
    weights_averged=FedAvg(weights)
    #load the global model with averaged weights
    global_model.load_state_dict(weights_averged)

    #checking the global validation loss
    global_model.eval()
    running_loss_val=0.0
    batchcount=0
    num_elements=0
    total_corr=0
    for batch in dataloader_global_val:
      batchcount+=1
      inputs=batch[0].to(device)
      num_elements+=inputs.shape[0]
      labels=batch[1].to(device)
      with torch.set_grad_enabled(False):
        outputs = global_model(inputs)
        validloss = criterion(outputs, labels)
        running_loss_val += validloss.item()*inputs.size(0)	
        total_corr+= torch.eq(torch.argmax(act(outputs),dim=1), labels).sum().item()
    acc_metric_val=float(total_corr)/num_elements
    #epoch_loss_train = running_loss_train / len(datasetTrain)
    epoch_loss_val = running_loss_val / len(datasetValid)

    print('Global Val Loss: {:.4f} Global Val Accuracy : {:.4f}'.format(epoch_loss_val,acc_metric_val))
    

	# Save model w/ lowest global_val loss
    if epoch_loss_val < bestvalloss:
    	torch.save(global_model.state_dict(), modelname+'.pth.tar')
    	bestvalloss = epoch_loss_val
