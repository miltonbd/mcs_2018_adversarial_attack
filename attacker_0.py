import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from attacker_model import *
from attacker import Attacker

args['cuda']='1'
args['save_root']='./changed_images_1/'
args['eps']=16
args['decay']=1
args['ssim_thr']=SSIM_THR
args['max_iter']=10000
args['datalist']='../data/pairs_list1.csv'


attacker = Attacker( transform, img2tensor, args)

img_pairs = pd.read_csv(args['datalist'])
for idx in tqdm(img_pairs.index.values):
    pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                 'target': img_pairs.loc[idx].target_imgs.split('|')}
    attacker.attack_method=attacker.MI_FGSM_L2
    attacker.attack(pair_dict)

