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

class Attacker():
    '''
    FGSM attacker: https://arxiv.org/pdf/1412.6572.pdf
    model -- white-box model for attack
    eps -- const * Clipped Noise
    ssim_thr -- min value for ssim compare
    transform -- img to tensor transform without CenterCrop and Scale
    '''
    def __init__(self, transform, img2tensor,args):
        logdir = './logs'
        if  os.path.exists(logdir):
            shutil.rmtree(logdir)
        os.makedirs(logdir)
        self.writer= SummaryWriter(logdir)
        self.eps=args['eps']
        self.ssim_thr = args['ssim_thr']
        self.max_iter = args['max_iter']
        self.decay=args['decay']
        self.transform = transform
        self.cropping = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.Scale(112)
                                      ])
        self.img2tensor = img2tensor
        self.args = args
        self.loss = nn.BCEWithLogitsLoss()
        self.target_descriptors=[]
        self.attack_method=""

        # if os.path.exists(args['save_root']):
        #     shutil.rmtree(args['save_root'])

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

        fmodel = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 1), num_classes=512)
        self.fmodel=fmodel
        target_img_names = attack_pairs['target']
        self.target_descriptors = np.ones((len(attack_pairs['target']), 512),
                                     dtype=np.float32)

        for idx, img_name in enumerate(target_img_names):
            img_name = os.path.join(self.args["root"], img_name)
            img = Image.open(img_name)
            tensor = self.transform(img).unsqueeze(0)
            if self.args["cuda"]:
                tensor = tensor.cuda(async=True)

            res = self.model(Variable(tensor, requires_grad=False)).data.cpu().numpy().squeeze()
            self.target_descriptors[idx] = res

        for img_name in attack_pairs['source']:
            #img is attacked
            if os.path.isfile(os.path.join(self.args["save_root"], img_name)):
                continue
            source_img_path=os.path.join(self.args['root'], img_name)
            attacked_img=self.attack_method(source_img_path)

            if not os.path.isdir(self.args['save_root']):
                os.makedirs(self.args['save_root'])
            attacked_img.save(os.path.join(self.args['save_root'], img_name.replace('.jpg', '.png')))

    # def FGSMAttack(self, source_img_path):
    #     img = Image.open(source_img_path)
    #     original_img = self.cropping(img)
    #
    #     transform = transforms.Compose([transforms.CenterCrop(224),
    #                                     transforms.Scale(112),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=MEAN, std=STD),
    #                                     ])
    #     tensor = transform(transform)
    #     input_var = Variable(tensor.unsqueeze(0).cuda(async=True),
    #                          requires_grad=True).type(torch.cuda.FloatTensor)
    #
    #     ssim_final=1
    #     attacked_img = original_img
    #     iter_passed=0
    #     for iter_number in range(self.max_iter):
    #         iter_passed=iter_number
    #         adv_noise = torch.zeros((3, 112, 112))
    #
    #         adv_noise = adv_noise.cuda(async=True)
    #
    #         for idx,target_descriptor in enumerate(self.target_descriptors):
    #             target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
    #                                   requires_grad=False)
    #             input_var.grad = None
    #             out = self.model(input_var)
    #             loss = self.loss(out, target_out)
    #             self.writer.add_scalar("loss of "+source_img_path,loss, iter_number*idx+idx)
    #             loss.backward()
    #             noise = self.eps * torch.sign(input_var.grad.data).squeeze()
    #             adv_noise = adv_noise + noise
    #
    #         self.writer.add_scalar("loss of " + source_img_path, loss, iter_number)
    #         input_var.data = input_var.data - adv_noise
    #         changed_img = self.tensor2img(input_var.data.cpu().squeeze())
    #
    #         # SSIM checking
    #         ssim = compare_ssim(np.array(original_img, dtype=np.float32),
    #                             np.array(changed_img, dtype=np.float32),
    #                             multichannel=True)
    #         ssim_final = ssim
    #         if ssim < self.ssim_thr:
    #             break
    #         else:
    #             attacked_img = changed_img
    #     print("ssim:{}, iter:{}".format(ssim_final,iter_passed))
    #     return attacked_img

    def M_DI_2_FGSM(self, source_img_path):
        img = Image.open(source_img_path)
        original_img = self.cropping(img)

        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.Scale(112),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEAN, std=STD),
                                        ])
        tensor = transform(img)
        input_var = Variable(tensor.unsqueeze(0).cuda(async=True),
                             requires_grad=True).type(torch.cuda.FloatTensor)
        input_var_clone=input_var.clone()
        eps_init = 0.01
        decay= 1
        attacked_img = original_img
        grads=0
        ssim_final = 1
        iter_passed = 0
        # todo apply l2
        # todo input augmentation and change them input DI input
        # todo add drop out
        # todo try other regulerizer
        # todo

        for iter_number in range(self.max_iter):
            iter_passed=iter_number
            adv_noise = torch.zeros((3, 112, 112))
            adv_noise = adv_noise.cuda(async=True)
            alpha = eps_init  # todo learning rate decay
            # eps=self.decayEps(eps_init,iter_number,self.max_iter)
            eps=eps_init
            criterion = torch.nn.MSELoss()
            # self.writer.add_scalar('eps for '+source_img_path,eps,iter_number)
            for idx,target_descriptor in enumerate(self.target_descriptors):
                target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
                                      requires_grad=False)

                input_var.grad = None
                self.model.cuda()
                out = self.model(input_var)

                loss =  criterion(out, target_out)
                self.writer.add_scalar("loss of " + source_img_path, loss, iter_number * idx + idx)
                loss.backward()
                curr_grad=input_var.grad

                grads= decay * grads + curr_grad/curr_grad.abs().sum()

                #noise = alpha * torch.sign(grads).squeeze()   # L0 norm

                noise = alpha * grads/grads.abs().pow(2).sum()   # L2 norm

                adv_noise = adv_noise + noise.data

                #self.writer.add_scalar("noise " + source_img_path, noise, iter_number)

            input_var.data = input_var.data - adv_noise.cuda()
            changed_img = self.tensor2img(input_var.data.cpu().squeeze())


            # SSIM checking
            ssim = compare_ssim(np.array(original_img, dtype=np.float32),
                                np.array(changed_img, dtype=np.float32),
                                multichannel=True)
            self.writer.add_scalar("ssim of " + source_img_path, ssim, iter_number)

            print("current ssim:{}, iter:{}".format(ssim, iter_passed))

            if ssim < self.ssim_thr:
                if int(ssim_final)==1:
                    print("\n\n\n\n ssim \n\n\n\n")
                    # input_var.data=input_var_clone.data
                    # input_var.grad=None
                    # continue
                break
            else:
                attacked_img = changed_img
            ssim_final=ssim
        print("final ssim:{}, iter:{}".format(ssim_final,iter_passed))
        return attacked_img

    # def MI_FGSM_L2(self, input_var, original_img):
    #     eps = self.eps
    #     decay = self.decay
    #     alpha = eps / 12
    #     attacked_img = original_img
    #     grads = 0
    #     ssim_final = 1
    #     iter_passed = 0
    #     for iter_number in range(self.max_iter):
    #         adv_noise = torch.zeros((3, 112, 112))
    #         adv_noise = adv_noise.cuda(async=True)
    #
    #         for target_descriptor in self.target_descriptors:
    #             target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
    #                                   requires_grad=False)
    #
    #             input_var.grad = None
    #             self.model.cuda()
    #             out = self.model(input_var)
    #             calc_loss = self.loss(out, target_out)
    #             calc_loss.backward()
    #             curr_grad = input_var.grad
    #             # if grads==0:
    #             #     grads=curr_grad
    #             grads = decay * grads + curr_grad / curr_grad.abs().sum()
    #
    #             noise = alpha * grads.abs().sum()/grads.abs().pow(2).sum()
    #             adv_noise = adv_noise + noise.data
    #
    #         input_var.data = input_var.data - adv_noise.cuda()
    #         changed_img = self.tensor2img(input_var.data.cpu().squeeze())
    #
    #         # SSIM checking
    #         ssim = compare_ssim(np.array(original_img, dtype=np.float32),
    #                             np.array(changed_img, dtype=np.float32),
    #                             multichannel=True)
    #         if ssim < self.ssim_thr:
    #             break
    #         else:
    #             attacked_img = changed_img
    #     print("final ssim:{}, iter:{}".format(ssim_final,iter_passed))
    #     return attacked_img
    #
    # def DI2_MI_FGSM(self, input_var, original_img):
    #     p=.4
    #     eps = self.eps
    #     decay= self.decay
    #     alpha=2/255
    #     attacked_img = original_img
    #     grads=0
    #     for iter_number in tqdm(range(self.max_iter)):
    #         adv_noise = torch.zeros((3, 112, 112))
    #         adv_noise = adv_noise.cuda(async=True)
    #
    #         for target_descriptor in self.target_descriptors:
    #             target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
    #                                   requires_grad=False)
    #
    #             input_var.grad = None
    #             self.model.cuda()
    #             out = self.model(input_var)
    #             calc_loss= F.cross_entropy(out, target_out)
    #             calc_loss.backward()
    #             curr_grad=input_var.grad
    #
    #             grads= decay * grads + curr_grad/curr_grad.abs().sum()
    #
    #             noise = alpha * torch.sign(grads).squeeze()
    #             adv_noise = adv_noise + noise.data
    #
    #         input_var.data = input_var.data - adv_noise.cuda()
    #         changed_img = self.tensor2img(input_var.data.cpu().squeeze())
    #
    #         # SSIM checking
    #         ssim = compare_ssim(np.array(original_img, dtype=np.float32),
    #                             np.array(changed_img, dtype=np.float32),
    #                             multichannel=True)
    #         if ssim < self.ssim_thr:
    #             break
    #         else:
    #             attacked_img = changed_img
    #     return attacked_img
    #
    #
    # def DI2_MI_FGSM_L2(self, input_var, original_img):
    #     ssim_final=1
    #     iter_passed=0
    #     eps = self.eps
    #     decay= self.decay
    #     alpha=eps/12
    #     attacked_img = original_img
    #     grads=0
    #     for iter_number in tqdm(range(self.max_iter)):
    #         iter_passed=iter_number
    #         adv_noise = torch.zeros((3, 112, 112))
    #         adv_noise = adv_noise.cuda(async=True)
    #
    #         for target_descriptor in self.target_descriptors:
    #             target_out = Variable(torch.from_numpy(target_descriptor).unsqueeze(0).cuda(async=True),
    #                                   requires_grad=False)
    #             input_var.grad = None
    #             self.model.cuda()
    #             out = self.model(input_var)
    #             calc_loss = self.loss(out, target_out)
    #             calc_loss.backward()
    #             curr_grad=input_var.grad
    #             # if grads==0:
    #             #     grads=curr_grad
    #             grads= decay * grads + curr_grad/curr_grad.abs().sum()
    #             noise = alpha * torch.sign(grads).squeeze()
    #             adv_noise = adv_noise + noise.data
    #
    #         input_var.data = input_var.data - adv_noise.cuda()
    #         changed_img = self.tensor2img(input_var.data.cpu().squeeze())
    #
    #         # SSIM checking
    #         ssim = compare_ssim(np.array(original_img, dtype=np.float32),
    #                             np.array(changed_img, dtype=np.float32),
    #                             multichannel=True)
    #         ssim_final = ssim
    #         if ssim < self.ssim_thr:
    #             break
    #         else:
    #             attacked_img = changed_img
    #     print("ssim:{}, iter:{}".format(ssim_final,iter_passed))
    #     return attacked_img

#