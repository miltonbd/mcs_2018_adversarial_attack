from attacker_model import *
from attacker import Attacker
import shutil

import pandas as pd
import os
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


args['save_root']='./changed_images_1/'

args['cuda']="1"
args['ssim_thr']=SSIM_THR
args['max_iter']=60

#
# if os.path.exists(args['save_root']):
#     shutil.rmtree(args['save_root'])



def start_gpu_thread(part_csv,logdir):
    args['logdir']=logdir
    args['datalist'] = part_csv
    if os.path.exists(args['logdir']):
        shutil.rmtree(args['logdir'])
        # exit(1)
    attacker = Attacker(transform, img2tensor, args)
    img_pairs = pd.read_csv(args['datalist'])

    for idx in tqdm(img_pairs.index.values):
        pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                     'target': img_pairs.loc[idx].target_imgs.split('|')}
        attacker.attack_method = attacker.M_DI_2_FGSM
        attacker.attack(pair_dict)