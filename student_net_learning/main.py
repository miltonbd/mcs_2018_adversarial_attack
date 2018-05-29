'''
Simple transfer learning.
Teacher model: Image descriptors from black-box model
Student model: VGG|ResNet|DenseNet
'''

from __future__ import print_function
import os
import argparse
import random

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
from dataset import ImageListDataset
from utils import progress_bar

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

#torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description='PyTorch student network training')

parser.add_argument('--lr',
                    default=0.001, 
                    type=float, 
                    help='learning rate')
parser.add_argument('--resume',
                    action='store_true', 
                    help='resume from checkpoint')
parser.add_argument('--optimizer',
                    type=str, 
                    help='optimizer type', 
                    default='adam')
parser.add_argument('--criterion',
                    type=str, 
                    help='criterion', 
                    default='MSE')
parser.add_argument('--root',
                    default='../data/',
                    type=str, 
                    help='data root path')
parser.add_argument('--datalist', 
                    default='../data/datalist/',
                    type=str, 
                    help='datalist path')
parser.add_argument('--batch_size', 
                    type=int, 
                    help='mini-batch size',
                    default=16)
parser.add_argument('--name',
                    required=True,
                    type=str, 
                    help='session name')
parser.add_argument('--log_dir_path',
                    default='./logs',
                    type=str, 
                    help='log directory path')
parser.add_argument('--epochs',
                    required=True,
                    type=int,
                    help='number of epochs')
parser.add_argument('--cuda',
                    action='store_true', 
                    help='use CUDA')
parser.add_argument('--model_name', 
                    type=str, 
                    help='model name', 
                    default='ResNet18')
parser.add_argument('--down_epoch', 
                    type=int, 
                    help='epoch number for lr * 1e-1', 
                    default=30)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    
    lr = args.lr * (0.1 ** (epoch//args.down_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    '''
    Train function for each epoch
    '''

    global net
    global trainloader
    global args
    global log_file
    global optimizer
    global criterion

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs, targets.squeeze()
        adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)

        optimizer.zero_grad()
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        curr_batch_loss = loss.data[0]
        train_loss += curr_batch_loss
        total += targets.size(0)

        log_file.write('train,{epoch},'\
                       '{batch},{loss:.3f}\n'.format(epoch=epoch, 
                                                     batch=batch_idx,
                                                     loss=curr_batch_loss))
        progress_bar(batch_idx, 
                     len(trainloader),
                     'Loss: {l:.3f}'.format(l = train_loss/(batch_idx+1)))

def validation(epoch):
    
    global net
    global valloader
    global best_loss
    global args
    global log_file

    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        inputs, targets = inputs, targets.squeeze()
        if args.cuda:
            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        curr_batch_loss = loss.data[0]
        val_loss += curr_batch_loss

        log_file.write('val,{epoch},'\
                       '{batch},{loss:.5f}\n'.format(epoch=epoch, 
                                                     batch=batch_idx,
                                                     loss=curr_batch_loss))
        progress_bar(batch_idx, 
                     len(valloader), 
                     'Loss: {l:.3f}'.format(l = val_loss/(batch_idx+1)))
    val_loss = val_loss/(batch_idx+1)
    if val_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict() if torch.cuda.device_count() <= 1 \
                                    else net.module.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
            'arguments': args
        }
        session_checkpoint = 'checkpoint/{name}/'.format(name=args.name)
        if not os.path.isdir(session_checkpoint):
            os.makedirs(session_checkpoint)
        torch.save(state, session_checkpoint + 'best_model_chkpt.t7')
        best_loss = val_loss

def main():
    global net
    global trainloader
    global valloader
    global best_loss
    global log_file
    global optimizer
    global criterion
    #initialize
    start_epoch = 0
    best_loss = np.finfo(np.float32).max

    #augmentation
    random_rotate_func = lambda x: x.rotate(random.randint(-15,15),
                                            resample=Image.BICUBIC)
    random_scale_func = lambda x: transforms.Scale(int(random.uniform(1.0,1.4)\
                                                   * max(x.size)))(x)
    gaus_blur_func = lambda x: x.filter(PIL.ImageFilter.GaussianBlur(radius=1))
    median_blur_func = lambda x: x.filter(PIL.ImageFilter.MedianFilter(size=3))

    #train preprocessing
    transform_train = transforms.Compose([
        transforms.Lambda(lambd=random_rotate_func),
        transforms.CenterCrop(224),
        transforms.Scale((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    #validation preprocessing
    transform_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Scale((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    print('==> Preparing data..')
    trainset = ImageListDataset(root=args.root, 
                                list_path=args.datalist, 
                                split='train', 
                                transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=8, 
                                              pin_memory=True)

    valset = ImageListDataset(root=args.root, 
                               list_path=args.datalist, 
                               split='val', 
                               transform=transform_val)

    valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=8, 
                                             pin_memory=True)

    # Create model
    net = None
    if args.model_name == 'ResNet18':
        net = ResNet18()
    elif args.model_name == 'ResNet34':
        net = ResNet34()
    elif args.model_name == 'ResNet50':
        net = ResNet50()
    elif args.model_name == 'DenseNet':
        net = DenseNet121()
    elif args.model_name == 'VGG11':
        net = VGG('VGG11')

    print('==> Building model..')

    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{0}/best_model_ckpt.t7'.format(args.name))
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1

    # Choosing of criterion
    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = None # Add your criterion

    # Choosing of optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Load on GPU
    if args.cuda:
        print ('==> Using CUDA')
        print (torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = net.cuda()
        cudnn.benchmark = True
        print ('==> model on GPU')
        criterion = criterion.cuda()
    else:
        print ('==> model on CPU')
    
    if not os.path.isdir(args.log_dir_path):
       os.makedirs(args.log_dir_path)
    log_file_path = os.path.join(args.log_dir_path, args.name + '.log')
    # logger file openning
    log_file = open(log_file_path, 'w')
    log_file.write('type,epoch,batch,loss,acc\n')

    print ('==> Model')
    print(net)

    try:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            validation(epoch)
        print ('==> Best loss: {0:.5f}'.format(best_loss))
    except Exception as e:
        print (e.message)
        log_file.write(e.message)
    finally:
        log_file.close()

if __name__ == '__main__':
    net = None
    trainloader = None
    valloader = None
    best_loss = None
    log_file = None
    optimizer = None
    criterion = None

    main()
