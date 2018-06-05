from attacker_model import *
from attacker import Attacker
import shutil
args['save_root']='./changed_images_1/'
args['logdir']='./logs/'


args['cuda']="1"
args['eps']=0.0001
args['decay']=1
args['ssim_thr']=SSIM_THR
args['max_iter']=1000


if os.path.exists(args['save_root']):
    shutil.rmtree(args['save_root'])

if os.path.exists(args['logdir']):
    shutil.rmtree(args['logdir'])

def start_gpu_thread(part_csv):
    args['datalist'] = part_csv
    attacker = Attacker(transform, img2tensor, args)
    img_pairs = pd.read_csv(args['datalist'])

    for idx in tqdm(img_pairs.index.values):
        pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                     'target': img_pairs.loc[idx].target_imgs.split('|')}
        attacker.attack_method = attacker.M_DI_2_FGSM
        attacker.attack(pair_dict)