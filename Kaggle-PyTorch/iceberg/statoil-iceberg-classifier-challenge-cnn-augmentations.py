

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
import pandas
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split
import logging
import numpy
import numpy as np

import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
import time
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import scipy

from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots
import tensorflow as tf 
print("tensorflow:" + tf.__version__)
import torch
import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
print('__Number CUDA Devices:', torch.cuda.device_count())

# !pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl
# !pip install torchvision 
# ! pip install cv2
# import cv2

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)

handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
import psutil
import torch
import gc    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

cpuStats()

use_cuda = torch.cuda.is_available()
# use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

# use_cuda=False
lgr.info("USE CUDA=" + str (use_cuda))

# dont! fix seed
# seed=17*19
# np.random.seed(seed)
# torch.manual_seed(seed)
# if use_cuda:
#     torch.cuda.manual_seed(seed)

base_path = os.path.join('..', 'input')


# # Concatenate and Reshape
# Here we load the data and then combine the two bands and recombine them into a single image/tensor for training

# In[2]:


# Data params
TARGET_VAR= 'target'
# BASE_FOLDER = '../input/'
# Data params
TARGET_VAR= 'target'
BASE_FOLDER = 'd:/db/data/ice/'

data = pd.read_json(BASE_FOLDER + '/train.json')

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

print (type(data))

# Suffle
import random
from datetime import datetime
random.seed(datetime.now())
# np.random.seed(datetime.now())
from sklearn.utils import shuffle
data = shuffle(data) # otherwise same validation set each time!
data= data.reindex(np.random.permutation(data.index))

band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
full_img = np.stack([band_1, band_2], axis=1)


from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler
# import cv2


# # Custom PyTorch Dataset to enable applying image Transforms
# - Since we have a non regular image type, a custom Dataset has to be written (adapted from:https://www.kaggle.com/supersp1234/tools-for-pytorch-transform and https://www.kaggle.com/heyt0ny/pytorch-custom-dataload-with-augmentaion)
# - This is required for enrichment 
# 

# In[3]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
import random
import PIL
from PIL import Image, ImageOps
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

batch_size=128   

class IcebergCustomDataSet(Dataset):
    """total dataset."""

    def __init__(self, data, labels,transform=None):
        self.data= data
        self.labels = labels
        self.transform = transform        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = image.astype(float)/255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).float()
               }
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']
        
        if random.random() < 0.5:
            image=np.flip(image,1)
        
        return {'image': image, 'labels': labels}
    
class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']
        
        if random.random() < 0.5:
            image=np.flip(image,0)
        
        return {'image': image, 'labels': labels} 

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        img=tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels']}  

from random import randrange    
random.seed(datetime.now()) # re seed 

X_train,X_val,y_train,y_val=train_test_split(full_img,data['is_iceberg'].values,
                                                   test_size=0.22, 
                                                   random_state=randrange(50000))

train_dataset = IcebergCustomDataSet(X_train, y_train, transform=transforms.Compose([
                                                              RandomHorizontalFlip(), 
                                                              RandomVerticallFlip(), 
                                                              ToTensor(), 
                                                              ])) 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=1)

val_dataset = IcebergCustomDataSet(X_val, y_val, 
                                transform=transforms.Compose([RandomHorizontalFlip(), 
                                                              RandomVerticallFlip(), 
                                                              ToTensor(), 
                                                              ])) 
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=1)

print (train_loader)
print (val_loader)    


# # Train/Validation split 
# (Not currently in use old version that did not involve image transforms)

# In[4]:



# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)        
#     print(x_data_np.shape)
#     print(type(x_data_np))

    if use_cuda:
#         lgr.info ("Using the GPU")    
        X_tensor = (torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    
    else:
#         lgr.info ("Using the CPU")
        X_tensor = (torch.from_numpy(x_data_np)) # Note the conversion for pytorch
        
#     print((X_tensor.shape)) # torch.Size([108405, 29])
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):    
    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!
#     print(y_data_np.shape)
#     print(type(y_data_np))

    if use_cuda:
#         lgr.info ("Using the GPU")            
    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        
    else:
#         lgr.info ("Using the CPU")        
    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        

#     print(type(Y_tensor)) # should be 'torch.cuda.FloatTensor'
#     print(y_data_np.shape)
#     print(type(y_data_np))    
    return Y_tensor

# class FullTrainningDataset(torch.utils.data.Dataset):
#     def __init__(self, full_ds, offset, length):
#         self.full_ds = full_ds
#         self.offset = offset
#         self.length = length
#         assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")
#         super(FullTrainningDataset, self).__init__()
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, i):
#         tItem=self.full_ds[i+self.offset]
#         img, label=tItem
#         return img, label
#
# validationRatio=0.11
#
# def trainTestSplit(dataset, val_share=validationRatio):
#     val_offset = int(len(dataset)*(1-val_share))
#     print ("Offest:" + str(val_offset))
#     return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset,
#                                             val_offset, len(dataset)-val_offset)

# # train_imgs = torch.from_numpy(full_img_tr).float()
# train_imgs=XnumpyToTensor (full_img)
# train_targets = YnumpyToTensor(data['is_iceberg'].values)
# dset_train = TensorDataset(train_imgs, train_targets)

# train_ds, val_ds = trainTestSplit(dset_train)

# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,num_workers=1)
# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)




# # CNN

# In[5]:


import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
dropout = [0.2, 0.4, 0.6, 0.7, 0.8]

class CNNClassifier(torch.nn.Module):
    def __init__(self, img_size, img_ch, kernel_size, pool_size, n_out):
        super(CNNClassifier, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_out = n_out
        self.sig=torch.nn.Sigmoid()
        self.all_losses = []
        self.val_losses = []
        
        self.build_model()
        print (self)
    # end constructor


    def build_model(self):
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.img_ch, 64, kernel_size=self.kernel_size, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout2d(p=dropout[0]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 2, kernel_size=self.kernel_size, padding=2),
            torch.nn.BatchNorm2d(2),
            torch.nn.Dropout2d(p=dropout[1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )                              

        self.fc = torch.nn.Linear(5184, self.n_out)
#         self.fc = torch.nn.Linear(int(self.img_size[0]/4)*int(self.img_size[1]/4)*32, self.n_out)        
        self.criterion = torch.nn.BCELoss()
        if use_cuda:
            self.criterion.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model

    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        # x= self.conv3(x)
        # x= self.conv4(x)
        x= self.shrink(x)
        x= self.fc(x)
        return self.sig(x)
    # end method forward

    def shrink(self, X):
        return X.view(X.size(0), -1)
    # end method flatten

    def fit(self,loader, num_epochs, batch_size):               
        self.train()
        for epoch in range(num_epochs):
            self.train()
            print('Epoch {}'.format(epoch + 1))
            print('*' * 5 + ':')
            running_loss = 0.0
            running_acc = 0.0            
    
            for i, dict_ in enumerate(loader):
                images  = dict_['image']
                target  = dict_['labels']
#                 self.train()
                inputs = torch.autograd.Variable(images)
                labels = torch.autograd.Variable(target)

                if use_cuda:
                    inputs.cuda()
                    labels.cuda()

                preds = self.forward(inputs)            # cnn output
                loss = self.criterion(preds, labels)    # cross entropy loss
                running_loss += loss.data[0] * labels.size(0)
                self.optimizer.zero_grad()              # clear gradients for this training step
                loss.backward()                         # backpropagation, compute gradients
                self.optimizer.step()                   # apply gradients
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == target.numpy()).mean()
                if (i+1) % 5 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                           %(epoch+1, num_epochs, i+1, 
                             int(len(train_dataset)/batch_size), loss.data[0], acc))                                                                            

        torch.save(self.state_dict(), './cnn.pth')
    # end method fit
    
    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
    
   
    # end class CNNClassifier


# # Train
def LeavOneOutValidation(val_loader):
    print('Leave one out VALIDATION ...')
    model = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out)
    if use_cuda:
        model.cuda()
    # .. to load your previously training model:
    model.load_state_dict(torch.load('./cnn.pth'))
    val_losses = []
    model.eval()

    print(val_loader)

    eval_loss = 0
    for data in val_loader:
        img = data['image']
        label = data['labels']
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
        if use_cuda:
            img.cuda()
            label.cuda()
        out = model(img)
        loss = model.criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)

    print('Leave one out VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_dataset))))
    val_losses.append(eval_loss / (len(val_dataset)))
    print()
    print()

img_size = (75,75)
img_ch = 2
kernel_size = 5
pool_size = 2
n_out = 1
n_epoch = 25

if __name__ == '__main__':
    cnn = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out)
    cnn.fit(train_loader,n_epoch, batch_size)
    LeavOneOutValidation(val_loader)

# # In[ ]:
#
#
# # kFoldValidation(10)
#
#
# # # Make Predictions
# # Here we make predictions on the output and export the CSV so we can submit
#
# # In[ ]:
#
#
# # load the model
# model=torch.load('./cnn.pth')
# model = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out)
#
# # .. to load your previously training model:
# model.load_state_dict(torch.load('./cnn.pth'))
# print (model)
#
# df_test_set = pd.read_json('../input/test.json')
#
# df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
# df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
# df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
#
# df_test_set.head(3)
#
#
# print (df_test_set.shape)
# columns = ['id', 'is_iceberg']
# df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
# # df_pred.id.astype(int)
#
# for index, row in df_test_set.iterrows():
#     rwo_no_id=row.drop('id')
#     band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
#     band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
#     full_img_test = np.stack([band_1_test, band_2_test], axis=1)
#
#     x_data_np = np.array(full_img_test, dtype=np.float32)
#     if use_cuda:
#         X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch
#     else:
#         X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch
#
# #     X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors
#     predicted_val = (model(X_tensor_test).data).float() # probabilities
#     p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float
#
#     df_pred = df_pred.append({'id':row['id'], 'is_iceberg':p_test},ignore_index=True)
# #     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)
#
# df_pred.head(5)
#
#
# def savePred(df_pred):
# #     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))
# #     csv_path = 'pred_{}_{}.csv'.format(loss, (str(time.time())))
#     csv_path='sample_submission.csv'
#     df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
#     print (csv_path)
#
# savePred (df_pred)
#
