import foolbox
import torch
import numpy as np
from student_net_learning.models.densenet import densenet201
from foolbox.criteria import TargetClass
from torchvision import transforms
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage.measure import compare_ssim
from torch.autograd import Variable
from foolbox.criteria import TargetClassProbability
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SSIM_THR = 0.95

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

REVERSE_MEAN = [-0.485, -0.456, -0.406]
REVERSE_STD = [1/0.229, 1/0.224, 1/0.225]


args = {
    'root':'../data/imgs/',
    'save_root':'./baseline1/',
    'datalist':'../data/pairs_list.csv',
    'model_name':'DenseNet',
    'checkpoint_path':'./student_net_learning/checkpoint/DenseNet/best_model_chkpt.t7',
    'cuda':'0'
}


pair_imgs_dir='../data/imgs'

transform = transforms.Compose([transforms.CenterCrop(224),transforms.Scale(112),transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),
            ])

def reverse_normalize(tensor, mean, std):
    '''reverese normalize to convert tensor -> PIL Image'''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def tensor2img(tensor, on_cuda=True):
    tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
    # clipping
    tensor[tensor > 1] = 1
    tensor[tensor < 0] = 0
    tensor = tensor.squeeze(0)
    if on_cuda:
        tensor = tensor.cpu()
    return transforms.ToPILImage()(tensor)


torchmodel = densenet201(pretrained=True)
checkpoint = torch.load("./student_net_learning/checkpoint/DenseNet/best_model_chkpt.t7")
torchmodel.load_state_dict(checkpoint['net'])

torchmodel.cuda()
torchmodel.eval()

fmodel = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 255), num_classes=512)

img_pairs = pd.read_csv("../data/pairs_list.csv")

# attack = foolbox.attacks.FGSM(fmodel,criterion=TargetClass())

from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClass
cr=TargetClass(11)

attack = LBFGSAttack(fmodel,criterion=cr)


def attack( attack_pairs):
    '''
    Args:
        attack_pairs (dict) - id pair, 'source': 5 imgs,
                                       'target': 5 imgs
    '''
    # print(attack_pairs)
    target_img_names = attack_pairs['target']
    target_descriptors = np.ones((len(attack_pairs['target']), 512),
                                 dtype=np.float32)

    for idx, img_name in enumerate(target_img_names):
        img_name = os.path.join(args["root"], img_name)
        img = Image.open(img_name)
        tensor = transform(img).unsqueeze(0)
        if args["cuda"]:
            tensor = tensor.cuda(async=True)

        res = self.model(Variable(tensor, requires_grad=False)).data.cpu().numpy().squeeze()
        target_descriptors[idx] = res

    # print ('TEST: target imgs are readed')
    for img_name in attack_pairs['source']:
        # print ('TEST: attack on image {0}'.format(img_name))

        # img is attacked
        if os.path.isfile(os.path.join(args["save_root"], img_name)):
            continue

        img = Image.open(os.path.join(args["root"], img_name))
        original_img = self.cropping(img)
        attacked_img = original_img
        tensor = transform(img)
        input_var = Variable(tensor.unsqueeze(0).cuda(async=True),
                             requires_grad=True)
        # print ('TEST: start iterations')
        # tick = time.time()
        for iter_number in tqdm(range(self.max_iter)):
            adv_noise = torch.zeros((3, 112, 112))

            adv_noise = adv_noise.cuda(async=True)

            for target_descriptor in target_descriptors:
                target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
                                      requires_grad=False)

                input_var.grad = None
                out = self.model(input_var)
                calc_loss = self.loss(out, target_out)
                calc_loss.backward()
                noise = self.eps * torch.sign(input_var.grad.data).squeeze()
                adv_noise = adv_noise + noise

            input_var.data = input_var.data - adv_noise
            changed_img = self.tensor2img(input_var.data.cpu().squeeze())

            # SSIM checking
            ssim = compare_ssim(np.array(original_img, dtype=np.float32),
                                np.array(changed_img, dtype=np.float32),
                                multichannel=True)
            if ssim < self.ssim_thr:
                break
            else:
                attacked_img = changed_img
        # tock = time.time()
        # print ('TEST: end iterations. Time: {0:.2f}sec'.format(tock - tick))

        if not os.path.isdir(self.args["save_root"]):
            os.makedirs(self.args["save_root"])
        attacked_img.save(os.path.join(self.args["save_root"], img_name.replace('.jpg', '.png')))

for idx in tqdm(img_pairs.index.values):
    for idx in tqdm(img_pairs.index.values):
        pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                     'target': img_pairs.loc[idx].target_imgs.split('|')}

        attack(pair_dict)
    break
    # adversarial = attack(image,1)

# print(np.sum(adversarial-image))
# image=image1.transpose((2, 0, 1)).copy()

#
# import matplotlib.pyplot as plt
#
# plt.figure()
#
# plt.subplot(1, 3, 1)
# plt.title('Original')
# plt.imshow(image1 / 255)  # division by 255 to convert [0, 255] to [0, 1]
# plt.axis('off')
#
# adversarial1=adversarial.transpose((1,2,0))
# plt.subplot(1, 3, 2)
# plt.title('Adversarial')
# plt.imshow( adversarial1 / 255)  # division by 255 to convert [0, 255] to [0, 1]
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.title('Difference')
# difference = image1-adversarial1
# print(difference.sum())
# plt.imshow(difference)
# plt.axis('off')
#
# plt.show()