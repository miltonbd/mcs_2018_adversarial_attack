from attacker_model import *
from attacker import Attacker
import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args['save_root']='./changed_images_0/'

attacker = Attacker(model, eps=1e-2, ssim_thr=SSIM_THR, transform=transform, img2tensor=img2tensor, args=args, max_iter=10000)
img_pairs = pd.read_csv(args['datalist'])
for idx in tqdm(img_pairs.index.values):
    pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                 'target': img_pairs.loc[idx].target_imgs.split('|')}
    attacker.attack_method=attacker.FGSMAttack
    attacker.attack(pair_dict)

