import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

label1 = [i for i in range(0,5)]
label2 = [i for i in range(5,10)]
label3 = [i for i in range(10,15)]

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    WC = os.listdir(root)  # Three working conditions WC0:35Hz12kN WC1:37.5Hz11kN WC2:40Hz10kN

    datasetname1 = os.listdir(os.path.join(root, WC[0]))
    datasetname2 = os.listdir(os.path.join(root, WC[1]))
    datasetname3 = os.listdir(os.path.join(root, WC[2]))
    data = []
    lab =[]
    for i in tqdm(range(len(datasetname1))):
        files = os.listdir(os.path.join('/tmp',root,WC[0],datasetname1[i]))
        for ii in [-4,-3,-2,-1]: #Take the data of the last three CSV files
            path1 = os.path.join('/tmp',root,WC[0],datasetname1[i],files[ii])
            data1, lab1 = data_load(path1,label=label1[i])
            data += data1
            lab +=lab1

    for j in tqdm(range(len(datasetname2))):
        files = os.listdir(os.path.join('/tmp',root,WC[1],datasetname2[i]))
        for jj in [-4,-3, -2, -1]:
            path2 = os.path.join('/tmp',root,WC[1],datasetname2[i],files[jj])
            data2, lab2 = data_load(path2,label=label2[j])
            data += data2
            lab += lab2

    for k in tqdm(range(len(datasetname3))):
        files = os.listdir(os.path.join('/tmp',root,WC[2],datasetname3[i]))
        for kk in [-4,-3, -2, -1]:
            path3 = os.path.join('/tmp',root,WC[2],datasetname3[i],files[kk])
            data3, lab3 = data_load(path3,label=label3[k])
            data += data3
            lab += lab3

    return [data, lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"]
    fl = fl.values
    fl = fl.reshape(-1,1)
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
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
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
class XJTU(object):
    num_classes = 15
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
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes= 15)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset



