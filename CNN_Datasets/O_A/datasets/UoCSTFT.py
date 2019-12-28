import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
import pickle
from scipy import signal
from sklearn.model_selection import train_test_split

signal_size = 1024

dataname = ["DataForClassification_Stage0.mat","DataForClassification_TimeDomain.mat"]

#label
label=[0,1,2,3,4,5,6,7,8]   #The data is labeled 0-8,they are {‘healthy’,‘missing’,‘crack’,‘spall’,‘chip5a’,‘chip4a’,‘chip3a’,‘chip2a’,‘chip1a’}

def  STFT(fl):
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)
    return img

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    path = os.path.join('/tmp',root,dataname[1])
    data = loadmat(path)
    data = data['AccTimeDomain']
    da = []
    lab = []

    start,end=0,104   #Number of samples per type=104
    i=0
    while end <= data.shape[1]:
        data1 = data[:,start:end]
        data1 = data1.reshape(-1,1)
        da1, lab1= data_load(data1,label=label[i])
        da +=da1
        lab +=lab1
        start += 104
        end += 104
        i += 1

    return [da,lab]

def data_load(fl,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = fl.reshape(-1,)

    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        imgs = STFT(x)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
        ReSize(size=10.0),
        Reshape(),
        Normalize(normlize_type),
        RandomScale(),
        RandomCrop(),
        Retype(),
    ]),
        'val': Compose([
        ReSize(size=10.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ])
    }
    return transforms[dataset_type]

def train_test_split_order(data_pd, test_size=0.8, num_classes=10):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True)
    return train_pd,val_pd
#--------------------------------------------------------------------------------------------------------------------
class UoCSTFT(object):
    num_classes = 9
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.data_dir, test)
            with open(os.path.join(self.data_dir, "UoCSTFT.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes= 9)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset