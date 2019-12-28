import os
import numpy as np 
import pandas as pd
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
from scipy import signal
from sklearn.model_selection import train_test_split

signal_size = 1024


#Three working conditions
WC1 = ["ib600_2.csv","n600_3_2.csv","ob600_2.csv","tb600_2.csv"]
WC2 = ["ib800_2.csv","n800_3_2.csv","ob800_2.csv","tb800_2.csv"]
WC3 = ["ib1000_2.csv","n1000_3_2.csv","ob1000_2.csv","tb1000_2.csv"]

label1 = [i for i in range(0,4)]
label2 = [i for i in range(4,8)]
label3 = [i for i in range(8,12)]

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
    
    data = []
    lab =[]
    for i in tqdm(range(len(WC1))):
        path1 = os.path.join('/tmp',root,WC1[i])
        data1, lab1 = data_load(path1,label=label1[i])
        data += data1
        lab +=lab1

    for j in tqdm(range(len(WC2))):
        path2 = os.path.join('/tmp',root,WC2[j])
        data2, lab2 = data_load(path2,label=label2[j])
        data += data2
        lab += lab2

    for k in tqdm(range(len(WC3))):
        path3 = os.path.join('/tmp',root,WC3[k])
        data3, lab3 = data_load(path3,label=label3[k])
        data += data3
        lab += lab3

    return [data, lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]/5:
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
class JNUSTFT(object):
    num_classes = 12
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
            with open(os.path.join(self.data_dir, "JNUSTFT.pkl"), 'wb') as fo:
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


