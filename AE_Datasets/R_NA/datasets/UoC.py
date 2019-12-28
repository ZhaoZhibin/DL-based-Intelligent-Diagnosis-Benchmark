import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024


dataname = ["DataForClassification_Stage0.mat","DataForClassification_TimeDomain.mat"]
#label
label=[0,1,2,3,4,5,6,7,8]   #The data is labeled 0-8,they are {‘healthy’,‘missing’,‘crack’,‘spall’,‘chip5a’,‘chip4a’,‘chip3a’,‘chip2a’,‘chip1a’}

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
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]
#--------------------------------------------------------------------------------------------------------------------
class UoC(object):
    num_classes = 9
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        list_data = get_files(self.data_dir, test)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset

