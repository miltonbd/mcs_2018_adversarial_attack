import os
import torch.nn.functional as F
from attacker_model import *
import foolbox
from student_net_learning.models.densenet import densenet201
import scipy
import attacks
import imageio
from attacks import *
import shutil
import torch.backends.cudnn as cudnn
from torch.nn.functional import dropout
from tensorboardX import SummaryWriter

from imgaug import augmenters as iaa
import imgaug as ia

from torch.utils.data.dataset import Dataset


def augment_images(sources_list):
    source_images = []
    original_images = []
    for source_img in sources_list:
        source_img_path = os.path.join(args['root'], source_img)
        img = Image.open(source_img_path)
        crop_rectangle = (13, 13, 224, 224)
        cropped_im = img.crop(crop_rectangle)
        resized = cropped_im.resize((112, 112), Image.ANTIALIAS)
        source_images.append(np.asarray(resized))
        original_image=np.asarray(resized).copy()
        original_images.append(original_image)
    source_images = np.asarray(source_images)
    original_images =np.asarray(original_images)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.05)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),

        # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # rotate=(-25, 25),
            shear=(-4, 4)
        ),
        iaa.Grayscale(alpha=(0.0, 1.0))
    ], random_order=True)  # apply augmenters in random order

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    images = np.asarray(source_images).astype(np.float32)
    images = np.repeat(images, 3, 0)
    # print(images.shape)
    images_aug = seq.augment_images(images)
    images_aug_arr = np.asarray(images_aug)
    images_aug_arr = np.concatenate((original_images, images_aug_arr), 0)
    # print(images_aug_arr.shape)
    return  (original_images,images_aug_arr)

class MultiTransformDataset(Dataset):
    def __init__(self,  sources, targets):
        self.transforms = transforms
        self.sources_original, self.sources = augment_images(sources)
        self.targets_original, self.targets = augment_images(targets)
        # print("Total Train source, targets:{} ".format(self.sources.shape[0]))

    def __getitem__(self, idx):
        img_source, img_target = self.sources[idx].transpose((2,0,1)),self.targets[idx].transpose((2,0,1))
        return (img_source, img_target)

    def __len__(self):
        return len(self.sources)


class Attacker():
    '''
    FGSM attacker: https://arxiv.org/pdf/1412.6572.pdf
    model -- white-box model for attack
    eps -- const * Clipped Noise
    ssim_thr -- min value for ssim compare
    transform -- img to tensor transform without CenterCrop and Scale
    '''
    def __init__(self, transform, img2tensor,args):
        self.writer= SummaryWriter(args['logdir'])
        self.ssim_thr = args['ssim_thr']
        self.max_iter = args['max_iter']
        self.transform = transform
        self.img2tensor = img2tensor
        self.args = args
        self.loss = nn.BCEWithLogitsLoss()
        self.target_descriptors=[]
        self.attack_method=""
        self.cropping = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Scale(112)
        ])
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Scale(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


    def tensor2img(self, tensor, on_cuda=True):
        tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
        # clipping
        tensor[tensor > 1] = 1
        tensor[tensor < 0] = 0
        tensor = tensor.squeeze(0)
        if on_cuda:
            tensor = tensor.cpu()
        return transforms.ToPILImage()(tensor)
    
    def get_SSIM(self, original_img, changed_img ):
        ssim = compare_ssim(np.array(original_img, dtype=np.float32),
                            np.array(changed_img, dtype=np.float32),
                            multichannel=True)
        return ssim

    def decayEps(self, init_eps, iter, max_iter):
        eps = init_eps * (1 - iter / max_iter) ** 2
        return init_eps

    def attack(self, attack_pairs):
        torchmodel = densenet201(pretrained=True)
        checkpoint = torch.load("./student_net_learning/checkpoint/DenseNet/best_model_chkpt.t7")
        torchmodel.load_state_dict(checkpoint['net'])

        torchmodel.cuda()
        cudnn.benchmark = True

        torchmodel.eval()
        self.model=torchmodel

        face_data_set=MultiTransformDataset(attack_pairs['source'], attack_pairs['target'])
        self.face_data_loader_desc = torch.utils.data.DataLoader(face_data_set, batch_size=25, shuffle=True,
                                    num_workers=2)

        sources_original = face_data_set.sources_original


        for batch_idx,  sources_img in enumerate(sources_original):
            sources_img=sources_original[batch_idx]
            img_name=attack_pairs['source'][batch_idx]
            save_path=os.path.join(self.args['save_root'], img_name.replace('.jpg', '.png'))
            if os.path.exists(save_path):
                continue
            attacked_img=self.attack_method(sources_img,img_name)
            if not os.path.isdir(self.args['save_root']):
                os.makedirs(self.args['save_root'])
            try:
                attacked_img.save(save_path)        # exit(1)
            except Exception:
                imageio.imwrite(save_path,attacked_img)

    def M_DI_2_FGSM(self, sources_img, source_image_name):
        source_img_path = os.path.join(args['root'], source_image_name)
        img = Image.open(source_img_path)
        original_img=self.cropping(img)
        tensor = self.transform(img)
        input_var =  Variable(tensor.unsqueeze(0).cuda(async=True),requires_grad=True)
        input_var_clone=input_var.clone()
        eps= 0.3
        decay= 1
        attacked_img = sources_img
        grads=0
        ssim_final = 1
        iter_passed = 0
        # todo try other regulerizer
        # todo pair augmentation, random erase

        for iter_number in range(self.max_iter):
            iter_passed=iter_number
            adv_noise = torch.zeros((3, 112, 112))
            adv_noise = adv_noise.cuda(async=True)
            alpha = eps/self.max_iter # todo learning rate decay
            # eps=self.decayEps(eps_init,iter_number,self.max_iter)
            criterion = torch.nn.MSELoss()
            criterion.cuda()
            # self.writer.add_scalar('eps for '+source_img_path,eps,iter_number)
            total_loss=0
            # print("each source has tragets: {}".format(self.target_descriptors.shape[0]))
            for batch_idx, (inputs, targets) in enumerate(self.face_data_loader_desc):
                self.target_descriptors = self.model(
                    Variable(targets, requires_grad=False).type(torch.FloatTensor).cuda()).data.cpu().numpy().reshape(
                    -1, 512)

            for idx in range(len(self.target_descriptors)):
                target=self.target_descriptors[idx]
                target_out =  Variable(torch.from_numpy(target).unsqueeze(0).cuda(async=True),requires_grad=False)
                input_var.grad = None
                self.model.cuda()
                out = self.model(input_var)

                loss =  criterion(out, target_out)
                total_loss+=loss
                # self.writer.add_scalar("loss of " + source_img_path, loss, iter_number * idx + idx)
                loss.backward()
                curr_grad=input_var.grad
                # try different
                grads= decay * grads + curr_grad/curr_grad.abs().sum()

                # noise = alpha * torch.sign(grads).squeeze()   # L0 norm

                noise = alpha * grads/grads.abs().pow(2).sum()   # L2 norm

                adv_noise = (adv_noise + noise.data)

                #self.writer.add_scalar("noise " + source_img_path, noise, iter_number)

            input_var.data = input_var.data - adv_noise.cuda()
            changed_img = self.tensor2img(input_var.data.cpu().squeeze())


            # SSIM checking
            ssim = compare_ssim(np.array(original_img, dtype=np.float32),
                                np.array(changed_img, dtype=np.float32),
                                multichannel=True)
            self.writer.add_scalar("loss of " + str(source_image_name), total_loss, iter_number)
            self.writer.add_scalar("ssim of " + str(source_image_name), ssim, iter_number)
            if iter_number%5==0:
                print("old ssim: {}".format(ssim))

            if ssim < self.ssim_thr:
                if int(ssim_final)==1:
                    eps/=2
                    input_var.data=input_var_clone.data
                    input_var.grad=None
                    grads=0
                    continue
                elif  iter_number<3:
                    eps /= 1.9
                    input_var.data = input_var_clone.data
                    input_var.grad = None
                    grads = 0
                    continue
                break
            else:
                attacked_img = changed_img
            ssim_final=ssim
        if ssim < self.ssim_thr:
            attacked_img=original_img
        print("final ssim:{}, iter:{}".format(ssim_final,iter_passed))
        return attacked_img
