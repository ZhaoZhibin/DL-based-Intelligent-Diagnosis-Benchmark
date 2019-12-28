import os
import pandas as pd
import  numpy as np
from scipy.io import loadmat
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

signal_size = 1024


datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat", "174.mat", "189.mat", "201.mat", "213.mat", "250.mat",
             "262.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat",
              "263.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat",
              "264.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat",
              "265.mat"]  # 1730rpm

# label
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join('/tmp', root, datasetname[3])
    data_root2 = os.path.join('/tmp', root, datasetname[0])

    path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data, lab = data_load(path1, axisname=normalname[0],label=0)  # nThe label for normal data is 0

    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join('/tmp', data_root2, dataname1[i])

        data1, lab1 = data_load(path2, dataname1[i], label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        w = int(np.sqrt(signal_size))
        x = fl[start:end]
        imgs = x.reshape(w,w)
        data.append(imgs)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
        ReSize(size=1.0),
        Reshape(),
        Normalize(normlize_type),
        RandomScale(),
        RandomCrop(),
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

def train_test_split_order(data_pd, test_size=0.8, num_classes=10):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True)
    return train_pd,val_pd

class CWRUSlice(object):
    num_classes = 10
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
            with open(os.path.join(self.data_dir, "CWRUSlice.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes= 10)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset