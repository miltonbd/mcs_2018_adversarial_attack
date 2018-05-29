'''
Prepare data for student model learning
'''
import  MCS2018

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import glob

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
'''
parser.add_argument('--save_path',
                    required=True,
                    type=str,
                    help='path to save descriptors (.npy)')

parser.add_argument('--batch_size', 
                    type=int, 
                    help='mini-batch size',
                    default=16)
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

def main(args):
    net = MCS2018.Predictor(args.gpu_id)

    #img list is needed for descriptors order
    img_list = glob.glob(os.path.join(args.root, '*.jpg'))[:1000]
    #img_list = pd.read_csv(args.datalist).path.values
    descriptors = np.ones((len(img_list),512), dtype=np.float32)

    preprocessing = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Scale(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                ])

    for idx, img_name in tqdm(enumerate(img_list), total=len(img_list)):
        img = Image.open(img_name)
        img_arr = preprocessing(img).unsqueeze(0).numpy()
        res = net.submit(img_arr).squeeze()
        descriptors[idx] = res

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

    im_list_df = pd.DataFrame(img_list)
    # save directory/img_name.jpg
    im_list_df[0] = im_list_df[0].apply(lambda x: '/'.join(x.split('/')[-2:]))
    im_path = os.path.join(args.datalist_path, 
                           'im_{type}.txt'.format(type=args.datalist_type))
    im_list_df.to_csv(im_path, header=False, index=False)

    at_path = os.path.join(args.datalist_path, 
                           'at_{type}.npy'.format(type=args.datalist_type))
    np.save(at_path,descriptors)

if __name__ == '__main__':
    main(args)
