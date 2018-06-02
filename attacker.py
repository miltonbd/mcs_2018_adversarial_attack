import os

from attacker_model import *
import foolbox


class Attacker():
    '''
    FGSM attacker: https://arxiv.org/pdf/1412.6572.pdf
    model -- white-box model for attack
    eps -- const * Clipped Noise
    ssim_thr -- min value for ssim compare
    transform -- img to tensor transform without CenterCrop and Scale
    '''
    def __init__(self, model, eps, ssim_thr, transform, img2tensor,
                 args, max_iter=50):
        self.model = model
        self.model.eval()
        self.eps = eps
        self.ssim_thr = ssim_thr
        self.max_iter = max_iter
        self.transform = transform
        self.cropping = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.Scale(112)
                                      ])
        self.img2tensor = img2tensor
        self.args = args
        self.loss = nn.MSELoss()
        self.target_descriptors=[]
        self.attack_method=""

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

    def attack(self, attack_pairs):
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

            img = Image.open(os.path.join(self.args["root"], img_name))
            original_img = self.cropping(img)
            tensor = self.transform(img)
            input_var = Variable(tensor.unsqueeze(0).cuda(async=True),
                                 requires_grad=True)

            attacked_img=self.attack_method(input_var,original_img)

            if not os.path.isdir(self.args["save_root"]):
                os.makedirs(self.args["save_root"])
            attacked_img.save(os.path.join(self.args["save_root"], img_name.replace('.jpg', '.png')))

    def FGSMAttack(self, input_var, original_img):
        attacked_img = original_img
        for iter_number in tqdm(range(self.max_iter)):
            adv_noise = torch.zeros((3, 112, 112))

            adv_noise = adv_noise.cuda(async=True)

            for target_descriptor in self.target_descriptors:
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
        return attacked_img


    def foolbox(self, input_var, original_img):
        from foolbox.criteria import TargetClass
        from foolbox.attacks import LBFGSAttack

        target_class = 22
        criterion = TargetClass(target_class)
        fmodel = foolbox.models.PyTorchModel(self.model, bounds=(0, 255), num_classes=512)

        attacked_img = original_img
        for target_descriptor in self.target_descriptors:

            attack = LBFGSAttack(fmodel, criterion)

            image = input_var.data.squeeze(0).cpu().numpy()
            label = np.argmax(fmodel.predictions(image))

            adversarial = attack(image, label=label)


        input_var.data = input_var.data - adv_noise
        changed_img = self.tensor2img(input_var.data.cpu().squeeze())


        # if self.get_SSIM(original_img,changed_img) < self.ssim_thr:
        #     break
        # else:
        #     attacked_img = changed_img
        return attacked_img
