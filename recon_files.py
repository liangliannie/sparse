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


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

dic = unpickle('/home/liang/Desktop/imagenet/Imagenet64_val/val_data')
img = dic['data']
all_result = []
w = 64
for i, mmg in enumerate(img):
    print(i)
    reconimage = img[i].reshape(3, 64, 64)
    reconimage = 0.2989 * reconimage[0] + 0.5870 * reconimage[1] + 0.1140 * reconimage[2]
    reconimage = rescale(reconimage, scale=w / 64.0, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., max(reconimage.shape), endpoint=False)
    sinogram = radon(reconimage, theta=theta, circle=True)

    sinogram = np.expand_dims(sinogram, axis=0)
    reconimage = np.expand_dims(reconimage, axis=0)
    sample = {'sino': sinogram, 'img': reconimage}
    all_result.append(sample)

with open('/home/liang/Desktop/imagenet/Imagenet64_val/reconfiles.pkl', 'wb') as f:
    pickle.dump(all_result, f)
print('finish')