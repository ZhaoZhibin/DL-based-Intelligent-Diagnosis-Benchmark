
import numpy as np
import torch
import cv2
import random
from scipy.signal import resample
from PIL import Image
import scipy

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        seq = seq[np.newaxis, :, :]
        return seq

class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

class ReSize(object):
    def __init__(self, size=1):
        self.size = size
    def __call__(self, seq):
        seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)
        seq = seq / 255
        return seq

class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        return seq*scale_factor


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
            return seq*scale_factor

class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_height = seq.shape[1] - self.crop_len
            max_length = seq.shape[2] - self.crop_len
            random_height = np.random.randint(max_height)
            random_length = np.random.randint(max_length)
            seq[random_length:random_length+self.crop_len, random_height:random_height+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq