'''
Prepare data for student model learning
'''

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import  MCS2018
# import  MCS2018 as   MCS20181
# import  MCS2018 as   MCS20182
# import  MCS2018 as   MCS20183
# import  MCS2018 as   MCS20184

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import glob
from sklearn.model_selection import train_test_split

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='Prepare data for training student model')

parser.add_argument('--root',
                    required=True,
                    type=str, 
                    help='data root path')

parser.add_argument('--datalist_path',
                    required=True,
                    type=str, 
                    help='img datalist directory path')
parser.add_argument('--datalist_type',
                     required=True,
                     type=str,
                     help='(train|val)')

parser.add_argument('--size',
                    type=int,
                    help='Total size of data',
                    default=1000000)

parser.add_argument('--batch_size',
                    type=int,
                    help='mini-batch size',
                    default=100)
'''
parser.add_argument('--save_path',
                    required=True,
                    type=str,
                    help='path to save descriptors (.npy)')


'''
parser.add_argument('--gpu_id',
                    type=int,
                    default=-1,
                    help='GPU id, if you want to use GPU. For CPU gpu_id=-1')
args = parser.parse_args()

'''
def chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        # Create an index range for l of n items:
        yield arr[i:i+chunk_size]
'''

def save_train_val_fold(img_list,descriptors,datalist_type):
    im_list_df = pd.DataFrame(img_list)
    # save directory/img_name.jpg
    im_list_df[0] = im_list_df[0].apply(lambda x: '/'.join(x.split('/')[-2:]))

    im_path = os.path.join(args.datalist_path,
                           'im_{type}.txt'.format(type=datalist_type))
    im_list_df.to_csv(im_path, header=False, index=False)

    at_path = os.path.join(args.datalist_path,
                           'at_{type}.npy'.format(type=datalist_type))
    np.save(at_path,descriptors)




import threading
import time
def main(args):
    # net = MCS2018.Predictor(0)
    net1=MCS2018.Predictor(1)
    #img list is needed for descriptors order
    print ("Total count:{}".format(args.size));

    img_list = glob.glob(os.path.join(args.root, '*.jpg'))[:args.size]
    #img_list = pd.read_csv(args.datalist).path.values
    descriptors = np.ones((len(img_list),512), dtype=np.float32)

    total_steps=len(img_list)

    def make_data(img_name, idx):
        preprocessing = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Scale(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        img = Image.open(img_name)
        img_arr = preprocessing(img).unsqueeze(0).numpy()
        # if idx%2==0:
        #     res = net.submit(img_arr).squeeze()
        # else:
        res = net1.submit(img_arr).squeeze()

        descriptors[idx] = res

    for idx,img_name in tqdm(enumerate(img_list), total=total_steps):
        t=threading.Thread(target=make_data, args=(img_name,idx))
        t.start()
        # t.join()
        time.sleep(.00101)
    '''
    for idx, img_names in tqdm(enumerate(chunks(img_list, args.batch_size))):
        img_arr = np.ones((len(img_names), 3, 112, 112), dtype=np.float32)
        for jdx, img_name in enumerate(img_names):
            img = Image.open(os.path.join(args.root, img_name))
            img_arr[jdx] = preprocessing(img).numpy()

        res = net.submit(img_arr)
        descriptors[idx * args.batch_size:(idx + 1) * arsg.batch_size] = res
    '''

    if not os.path.isdir(args.datalist_path):
        os.makedirs(args.datalist_path)

    train_end=int(args.size * .8)
    print(type(train_end))
    print(descriptors.shape)
    train_imgs=img_list[:train_end]
    train_desc=descriptors[:train_end,:]
    save_train_val_fold(train_imgs,train_desc, 'train')
    save_train_val_fold(img_list[train_end:],descriptors[train_end:,:],'val')


if __name__ == '__main__':
    main(args)
