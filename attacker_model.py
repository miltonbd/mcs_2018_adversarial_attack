import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from skimage.measure import compare_ssim

SSIM_THR = 0.95

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

REVERSE_MEAN = [-0.485, -0.456, -0.406]
REVERSE_STD = [1/0.229, 1/0.224, 1/0.225]

img2tensor = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=MEAN, std=STD)
             ])

transform = transforms.Compose([transforms.CenterCrop(224),transforms.Scale(112),transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),
            ])
parser = argparse.ArgumentParser(description='PyTorch student network training')

args = {
    'root':'../data/imgs/',
    'model_name':'DenseNet',
    'checkpoint_path':'./student_net_learning/checkpoint/DenseNet/best_model_chkpt.t7'
}

def reverse_normalize(tensor, mean, std):
    '''reverese normalize to convert tensor -> PIL Image'''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def get_model():
    from student_net_learning.models.densenet import densenet201
    print('Loading DenseNet121')
    net = densenet201(pretrained=True)
    checkpoint = torch.load(args['checkpoint_path'])
    net.load_state_dict(checkpoint['net'])
    return net
