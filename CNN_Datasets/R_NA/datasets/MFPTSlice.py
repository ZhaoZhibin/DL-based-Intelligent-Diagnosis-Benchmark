import os
import numpy as np 
import pandas as pd
from scipy.io import loadmat
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
import pywt
from sklearn.model_selection import train_test_split

signal_size=1024

#label
label1 = [1,2,3,4,5,6,7]
label2 = [ 8,9,10,11,12,13,14]   #The failure data is labeled 1-14

def  CWT(lenth,data):
    scale=np.arange(1,lenth)
    cwtmatr, freqs = pywt.cwt(data, scale, 'mexh')
    return cwtmatr

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    m = os.listdir(root)
    datasetname = os.listdir(os.path.join("/tmp", root, m[0]))  # '1 - Three Baseline Conditions'
    # '2 - Three Outer Race Fault Conditions'
    # '3 - Seven More Outer Race Fault Conditions'
    # '4 - Seven Inner Race Fault Conditions'
    # '5 - Analyses',
    # '6 - Real World Examples
    # Generate a list of data
    dataset1 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[0]))  # 'Three Baseline Conditions'
    dataset2 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[2]))  # 'Seven More Outer Race Fault Conditions'
    dataset3 = os.listdir(os.path.join("/tmp", root, m[0], datasetname[3]))  # 'Seven Inner Race Fault Conditions'
    data_root1 = os.path.join('/tmp',root,m[0],datasetname[0])  #Path of Three Baseline Conditions
    data_root2 = os.path.join('/tmp',root,m[0],datasetname[2])  #Path of Seven More Outer Race Fault Conditions
    data_root3 = os.path.join('/tmp',root,m[0],datasetname[3])  #Path of Seven Inner Race Fault Conditions
    
    path1=os.path.join('/tmp',data_root1,dataset1[0])
    data, lab = data_load(path1,label=0)  #The label for normal data is 0

    for i in tqdm(range(len(dataset2))):
        path2=os.path.join('/tmp',data_root2,dataset2[i])
        data1, lab1 = data_load(path2,label=label1[i])
        data += data1
        lab += lab1

    for j in tqdm(range(len(dataset3))):
        path3=os.path.join('/tmp',data_root3,dataset3[j])

        data2, lab2  = data_load(path3,label=label2[j])
        data += data2
        lab += lab2

    return [data,lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    if label==0:
        fl = (loadmat(filename)["bearing"][0][0][1])     #Take out the data
    else:
        fl = (loadmat(filename)["bearing"][0][0][2])     #Take out the data
    fl = fl.reshape(-1,)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        w = int(np.sqrt(signal_size))
        x = fl[start:end]
        imgs = x.reshape(w,w)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data,lab
def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
    'train': Compose([
        ReSize(size=10.0),
        Reshape(),
        Normalize(normlize_type),
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

#--------------------------------------------------------------------------------------------------------------------
class MFPTSlice(object):
    num_classes = 15
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
            with open(os.path.join(self.data_dir, "MFPTSlice.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset


