import os
import numpy as np 
import pandas as pd
from itertools import islice 
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
import pywt
from sklearn.model_selection import train_test_split

signal_size=1024

#Data names of 5 bearing fault types under two working conditions
Bdata = ["ball_20_0.csv","comb_20_0.csv","health_20_0.csv","inner_20_0.csv","outer_20_0.csv","ball_30_2.csv","comb_30_2.csv","health_30_2.csv","inner_30_2.csv","outer_30_2.csv"]
label1 = [i for i in range(0,10)]
#Data names of 5 gear fault types under two working conditions
Gdata = ["Chipped_20_0.csv","Health_20_0.csv","Miss_20_0.csv","Root_20_0.csv","Surface_20_0.csv","Chipped_30_2.csv","Health_30_2.csv","Miss_30_2.csv","Root_30_2.csv","Surface_30_2.csv"]
labe12 = [i for i in range(10,20)]

def  CWT(lenth,data):
    scale=np.arange(1,lenth)
    cwtmatr, freqs = pywt.cwt(data, scale, 'mexh')
    return cwtmatr


#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    datasetname:List of  dataset
    '''
    datasetname = os.listdir(os.path.join(root, os.listdir(root)[2]))  # 0:bearingset, 2:gearset
    root1 = os.path.join("/tmp",root,os.listdir(root)[2],datasetname[0]) #Path of bearingset
    root2 = os.path.join("/tmp",root,os.listdir(root)[2],datasetname[2]) #Path of gearset
    data = []
    lab =[]
    for i in tqdm(range(len(Bdata))):
        path1 = os.path.join('/tmp',root1,Bdata[i])
        data1, lab1 = data_load(path1,dataname=Bdata[i],label=label1[i])
        data += data1
        lab +=lab1

    for j in tqdm(range(len(Gdata))):
        path2 = os.path.join('/tmp',root2,Gdata[j])
        data2, lab2 = data_load(path2,dataname=Gdata[j],label=labe12[j])
        data += data2
        lab += lab2

    return [data, lab]

def data_load(filename,dataname,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    f = open(filename,"r",encoding='gb18030',errors='ignore')
    fl=[]
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",",8)   #Separated by commas
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  #Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t",8)   #Separated by \t
            fl.append(eval(word[1]))   # Take a vibration signal in the x direction as input
    fl = np.array(fl)
    fl = fl.reshape(-1,)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]/10:
        w = int(np.sqrt(signal_size))
        x = fl[start:end]
        imgs = x.reshape(w,w)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
    'train': Compose([
        ReSize(size=1.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ]),
    'val': Compose([
        ReSize(size=1.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ])
}
    return transforms[dataset_type]

#--------------------------------------------------------------------------------------------------------------------
class SEUSlice(object):
    num_classes = 20
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
            with open(os.path.join(self.data_dir, "SEUSlice.pkl"), 'wb') as fo:
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


