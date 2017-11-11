# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

from __future__ import print_function
from __future__ import division

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)

import numpy
import numpy as np
import random
from datetime import datetime
from sklearn.utils import shuffle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


import pandas
import pandas as pd

import logging

handler = logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)

# !pip install psutil
import psutil
import os


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

cpuStats()

# use_cuda=False
lgr.info("USE CUDA=" + str(use_cuda))



# Data params
TARGET_VAR = 'target'

# BASE_FOLDER = '../input/'
BASE_FOLDER = 'd:/db/data/ice/'

batch_size = 128
global_epoches = 55
validationRatio = 0.11
LR = 0.0005
MOMENTUM = 0.95
if use_cuda:
    num_workers = 0 # for windows version of PyTorch which does not share GPU tensors
else:
    num_workers = 4

global_seed=999

def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)
    # print(x_data_np.shape)
    # print(type(x_data_np))

    if use_cuda:
        # lgr.info("Using the GPU")
        X_tensor = (torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
    else:
        # lgr.info("Using the CPU")
        X_tensor = (torch.from_numpy(x_data_np))  # Note the conversion for pytorch

    # print((X_tensor.shape))  # torch.Size([108405, 29])
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):
    y_data_np = y_data_np.reshape((y_data_np.shape[0], 1))  # Must be reshaped for PyTorch!
    # print(y_data_np.shape)
    # print(type(y_data_np))

    if use_cuda:
        # lgr.info("Using the GPU")
        #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float
    else:
        # lgr.info("Using the CPU")
        #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float

    # print(type(Y_tensor))  # should be 'torch.cuda.FloatTensor'
    # print(y_data_np.shape)
    # print(type(y_data_np))
    return Y_tensor


class IcebergCustomDataSet(Dataset):
    """total dataset."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx, :, :, :], 'labels': np.asarray([self.labels[idx]])}
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
        # image = image.transpose((2, 0, 1))
        image = image.astype(float) / 255
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
            image = np.flip(image, 1)

        return {'image': image, 'labels': labels}


class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.3:
            image = np.flip(image, 0)
        return {'image': image, 'labels': labels}


class RandomTranspose(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.7:
            image = np.transpose(image, 0)
        return {'image': image, 'labels': labels}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        img = tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels']}

class FullTrainningDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]

def trainTestSplit(dataset, val_share=validationRatio):
    val_offset = int(len(dataset) * (1 - val_share))
    # print("Offest:" + str(val_offset))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset,val_offset, len(dataset) - val_offset)





# def readSuffleData(seed=datetime.now()):
def readSuffleData(seed_num):
    fixSeed(seed_num)
    local_data = pd.read_json(BASE_FOLDER + '/train.json')

    local_data = shuffle(local_data)  # otherwise same validation set each time!
    local_data = local_data.reindex(np.random.permutation(local_data.index))
    # local_data = shuffle(local_data)  # otherwise same validation set each time!
    # local_data = local_data.reindex(np.random.permutation(local_data.index))
    local_data['band_1'] = local_data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['band_2'] = local_data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['inc_angle'] = pd.to_numeric(local_data['inc_angle'], errors='coerce')
    band_1 = np.concatenate([im for im in local_data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in local_data['band_2']]).reshape(-1, 75, 75)
    local_full_img = np.stack([band_1, band_2], axis=1)
    return local_data, local_full_img



def getTrainValLoaders():
    # global train_ds, val_ds, train_loader, val_loader
    train_imgs = XnumpyToTensor(full_img)
    train_targets = YnumpyToTensor(data['is_iceberg'].values)
    dset_train = TensorDataset(train_imgs, train_targets)
    local_train_ds, local_val_ds = trainTestSplit(dset_train)
    local_train_loader = torch.utils.data.DataLoader(local_train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    local_val_loader = torch.utils.data.DataLoader(local_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return local_train_loader, local_val_loader, local_train_ds, local_val_ds


def getCustomTrainValLoaders():
    # global train_ds, train_loader, val_loader
    from random import randrange
    X_train, X_val, y_train, y_val = train_test_split(full_img, data['is_iceberg'].values,
                                                      test_size=validationRatio,
                                                      random_state=global_seed)
    local_train_ds = IcebergCustomDataSet(X_train, y_train,
                                          transform=transforms.Compose([

                                            RandomHorizontalFlip(),
                                            RandomVerticallFlip(),
                                            ToTensor(),
                                            # Normalize(mean = [0.456],std =[0.229]),
                                        ]))
    local_val_ds = IcebergCustomDataSet(X_val, y_val,
                                       transform=transforms.Compose([
                                           RandomHorizontalFlip(),
                                           RandomVerticallFlip(),
                                           ToTensor(),
                                           # Normalize(mean=[0.456], std=[0.229]),
                                       ]))
    local_train_loader = DataLoader(dataset=local_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    local_val_loader = DataLoader(dataset=local_val_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # print(local_train_loader)
    # print(local_val_loader)
    return local_train_loader, local_val_loader, local_train_ds, local_val_ds



dropout = torch.nn.Dropout(p=0.30)
relu = torch.nn.LeakyReLU()
pool = nn.MaxPool2d(2, 2)

class ConvRes(nn.Module):
    def __init__(self, insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
            nn.BatchNorm2d(insize),
            # nn.Dropout(drate),
            torch.nn.Conv2d(insize, outsize, kernel_size=2, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.math(x)


class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(pool, pool),
        )
        self.avgpool = torch.nn.AvgPool2d(pool, pool)

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cnn1 = ConvCNN(2, 32, kernel_size=7, pool=4, avg=False)
        self.cnn2 = ConvCNN(32, 32, kernel_size=5, pool=2, avg=True)
        self.cnn3 = ConvCNN(32, 32, kernel_size=5, pool=2, avg=True)

        self.res1 = ConvRes(32, 64)

        self.features = nn.Sequential(
            self.cnn1, dropout,
            self.cnn2,
            self.cnn3,
            self.res1,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(1024, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sig(x)
        return x


class ResNetLike(nn.Module):
    def __init__(self, block, layers, num_channels=2, num_classes=1):
        self.inplanes = 32
        super(ResNetLike, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        # self.dropout2 = nn.Dropout2d(p=0.3)
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # self.dropout3 = nn.Dropout2d(p=0.3)
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64 , num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        # x = self.dropout2(x)
        # x = self.layer3(x)
        # x = self.dropout3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print (x.data.shape)
        x = self.fc(x)
        x = self.sig(x)

        return x

import math

def savePred(df_pred, val_score):
    csv_path = str(val_score) + '_sample_submission.csv'
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print(csv_path)

def generateSingleModel(model,num_epoches=global_epoches):

    loss_func = torch.nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)  # L2 regularization
    if use_cuda:
        # lgr.info("Using the GPU")
        model.cuda()
        loss_func.cuda()

    # lgr.info(optimizer)
    # lgr.info(loss_func)

    criterion = loss_func
    all_losses = []
    val_losses = []

    for epoch in range(num_epoches):
        # print('Epoch {}'.format(epoch + 1))
        # print('*' * 5 + ':')
        running_loss = 0.0
        running_acc = 0.0
        for i, row_data in enumerate(train_loader, 1):
            img, label = row_data
            # img = row_data['image']
            # label = row_data['labels']
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(label)  # RuntimeError: expected CPU tensor (got CUDA tensor)

            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #     if i % 10 == 0:
        #         all_losses.append(running_loss / (batch_size * i))
        #         print('[{}/{}] Loss: {:.6f}'.format(
        #             epoch + 1, num_epoches, running_loss / (batch_size * i),
        #             running_acc / (batch_size * i)))
        #
        # print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_ds))))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for row_data in val_loader:
            img, label = row_data
            # img = row_data['image']
            # label = row_data['labels']
            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True), Variable(label.cuda(async=True),volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)

        val_losses.append(eval_loss / (len(val_ds)))
        # print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))
        # print()

    print('TRAIN Loss: {:.6f}'.format(running_loss / (len(train_ds))))
    print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))
    val_result = '{:.6f}'.format(eval_loss / (len(val_ds)))
    # torch.save(model.state_dict(), './pth/' + val_result + '_cnn.pth')

    return model, val_result


def testModel():
    df_test_set = pd.read_json(BASE_FOLDER + '/test.json')
    df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
    df_test_set.head(3)
    print(df_test_set.shape)
    columns = ['id', 'is_iceberg']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
    # df_pred.id.astype(int)
    for index, row in df_test_set.iterrows():
        rwo_no_id = row.drop('id')
        band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
        band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
        full_img_test = np.stack([band_1_test, band_2_test], axis=1)

        x_data_np = np.array(full_img_test, dtype=np.float32)
        if use_cuda:
            X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
        else:
            X_tensor_test = Variable(torch.from_numpy(x_data_np))  # Note the conversion for pytorch

        # X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors
        predicted_val = (model(X_tensor_test).data).float()  # probabilities
        p_test = predicted_val.cpu().numpy().item()  # otherwise we get an array, we need a single float

        df_pred = df_pred.append({'id': row['id'], 'is_iceberg': p_test}, ignore_index=True)

    return df_pred


if __name__ == '__main__':
    # simplenet = SimpleNet()
    # print(simplenet)
    # fix seed
    fixSeed(global_seed)

    for i in range (0 , 10):
        model = SimpleNet()
        # model = ResNetLike(BasicBlock, [1, 3, 3, 1], num_channels=2, num_classes=1)
        print ("Ensamble number:" + str(i))
        data, full_img = readSuffleData(seed_num=global_seed)
        train_loader, val_loader, train_ds, val_ds = getTrainValLoaders()
        # train_loader, val_loader, train_ds, val_ds = getCustomTrainValLoaders()
        model, val_result=generateSingleModel(model, num_epoches=55)

    # df_pred = testModel()
    # savePred(df_pred, val_result)