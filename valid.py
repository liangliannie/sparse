from __future__ import print_function, division
import os
from pytorch_msssim import MSSSIM
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data_dir
from skimage.transform import radon, rescale
import visdom
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch._jit_internal import weak_module, weak_script_method
import torch
import pickle
import unet


w = 64

testimage = imread(data_dir + "/phantom.png", as_gray=True)
testimage = rescale(testimage, scale=w/400.0, mode='reflect', multichannel=False)
theta = np.linspace(0., 180., max(testimage.shape), endpoint=False)
tesgram = radon(testimage, theta=theta, circle=True)

tesgram = np.expand_dims(tesgram, axis=0)
testimage = np.expand_dims(testimage, axis=0)
tesgram = np.expand_dims(tesgram, axis=0)
testimage = np.expand_dims(testimage, axis=0)
