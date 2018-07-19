import argparse
parser = argparse.ArgumentParser(description='PyTorch student network training')

parser.add_argument('--lr',default=0.0001,
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
                    default='/../../data/datalist/',
                    type=str,
                    help='datalist path')
parser.add_argument('--batch_size',
                    type=int,
                    help='mini-batch size',
                    default=150)

parser.add_argument('--log_dir_path',
                    default='./student_net_learning/logs',
                    type=str,
                    help='log directory path')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='number of epochs')
parser.add_argument('--cuda',
                    type=int,
                    default=1,
                    help='use CUDA')
parser.add_argument('--model_name',
                    type=str,
                    help='model name',
                    default='DenseNet')
parser.add_argument('--down_epoch',
                    type=int,
                    help='epoch number for lr * 1e-1',
                    default=30)