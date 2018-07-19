
from __future__ import print_function
import os
import argparse
import random
import sys
sys.path.append('/media/milton/ssd1/research/ai-artist')
from utils.functions import progress_bar

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import PIL

from models import *
from student_net_learning.dataset import ImageListDataset
# from utils import progress_bar


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# augmentation
random_rotate_func = lambda x: x.rotate(random.randint(-15, 15),
                                        resample=Image.BICUBIC)
random_scale_func = lambda x: transforms.Scale(int(random.uniform(1.0, 1.4) \
                                                   * max(x.size)))(x)
gaus_blur_func = lambda x: x.filter(PIL.ImageFilter.GaussianBlur(radius=1))
median_blur_func = lambda x: x.filter(PIL.ImageFilter.MedianFilter(size=3))

# train preprocessing
transform_train = transforms.Compose([
    transforms.Lambda(lambd=random_rotate_func),
    transforms.CenterCrop(224),
    transforms.Scale((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# validation preprocessing
transform_val = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Scale((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

print('==> Preparing data..')

def get_data(args):
    trainset = ImageListDataset(root=args.root,
                                list_path='/media/milton/ssd1/research/competitions/data/datalist/',
                                split='train',
                                transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)

    valset = ImageListDataset(root=args.root,
                              list_path='/media/milton/ssd1/research/competitions/data/datalist/',
                              split='val',
                              transform=transform_val)

    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True)
    return  trainloader,valloader

