'''
Simple transfer learning.
Teacher model: Image descriptors from black-box model
Student model: VGG|ResNet|DenseNet
'''

from __future__ import print_function
import os
import argparse
import random
import sys
sys.path.append('/media/milton/ssd1/research/ai-artist')
from utils.functions import progress_bar

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transformsd
from torch.autograd import Variable
from PIL import Image
import PIL

from models import *
from dataset import ImageListDataset
# from utils import progress_bar

from torchsummary import summary

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

#torch.set_default_tensor_type('torch.FloatTensor')
from arguments import *
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
    total = 0
    total_count=len(trainloader)
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
        step_loss=train_loss / (batch_idx + 1)
        log_file.add_scalar('Train Step Loss',step_loss, epoch*total_count+batch_idx)
        progress_bar(batch_idx,
                      total_count,
                      'Train Loss step: {l:.3f}'.format(l=step_loss))

    log_file.add_scalar('train loss',train_loss,epoch)


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
        progress_bar(batch_idx, 
                     len(valloader), 
                     'Validation Loss Step: {l:.3f}'.format(l = val_loss/(batch_idx+1)))
    log_file.add_scalar('Validation Loss',val_loss, epoch)
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
        print('Loading ResNet18')
        net = ResNet18()
    elif args.model_name == 'ResNet34':
        print('Loading ResNet34')
        net = ResNet34()
    elif args.model_name == 'ResNet50':
        # print('Loading pnasnet5large')
        # from student_net_learning.pretrainedmodels.models.pnasnet import pnasnet5large
        # net = pnasnet5large()
        # net = dpn131(pretrained=True)
        print("ResNET152")
        from classification.models.pytorch.resnet import resnet152
        net=resnet152(pretrained=True)
    elif args.model_name == 'DenseNet':
        print('Loading DenseNet121')
        net = DenseNet121()
    elif args.model_name == 'VGG11':
        print('Loading VGG11')
        net = VGG('VGG11')
    elif args.model_name == 'VGG19_BN':
        from student_net_learning.pretrainedmodels.models.pnasnet import pnasnet5large
        print('Loading VGG19_BN')
        net = pnasnet5large()

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
    log_file_path = os.path.join(args.log_dir_path, args.name )
    # logger file openning
    log_file = SummaryWriter(log_file_path)

    total=0

    for name, child in net.named_children():
        total+=1
    print("Total Layer:{}".format(total))

    ct = total

    for name2, params in child.named_parameters():
        if ct < 1:
            params.requires_grad = True
        else:
            params.requires_grad=False
        ct-=1

    print('==> Model')
    summary(net, (3, 112, 112))

    # Choosing of optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,eps=1)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    try:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            validation(epoch)
        print ('==> Best loss: {0:.5f}'.format(best_loss))
    except Exception as e:
        print (e.message)
    finally:
        pass
if __name__ == '__main__':
    net = None
    trainloader = None
    valloader = None
    best_loss = None
    log_file = None
    optimizer = None
    criterion = None

    main()
