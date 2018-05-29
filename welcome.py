import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import zipfile
import MCS2018
#import MCS2018_CPU as MCS2018 if you are using CPU only black box model
gpu_id = 0
net = MCS2018.Predictor(gpu_id)
from torchvision import transforms
from skimage.measure import compare_ssim as ssim
df = pd.read_csv('../data/pairs_list.csv')
df.head()
imgs_path = '../data/imgs/'
os.listdir(imgs_path)[:10]
for idx in df.index[:5]:
    source_imgs = df.loc[idx].source_imgs
    target_imgs = df.loc[idx].target_imgs
    plt.figure(figsize=(20, 5))
    for i, img_name in enumerate(source_imgs.split('|'), 1):
        img = Image.open(os.path.join(imgs_path, img_name))
        # plt.subplot(1, 10, i)
        # plt.title('S{}_Im{}'.format(idx, i))
        # plt.axis('off')
        # plt.imshow(img)

    for i, img_name in enumerate(target_imgs.split('|'), 1):
        img = Image.open(os.path.join(imgs_path, img_name))
        # plt.subplot(1, 10, i + 5)
        # plt.title('T{}_Im{}'.format(idx, i))
        # plt.axis('off')
        # plt.imshow(img)rebopot



def preprocess_img(img):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.Resize(112),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                    ])
    img_arr = preprocessing(img).unsqueeze(0).numpy()
    return img_arr
img_arr = preprocess_img(img)
img_arr.shape

img_descriptor = net.submit(img_arr).squeeze()
img_descriptor.shape


source_imgs = df.loc[0].source_imgs
target_imgs = df.loc[0].target_imgs


source_desc=[]
target_desc=[]
for img_name in source_imgs.split('|'):
    img = Image.open(os.path.join(imgs_path,img_name))
    img_arr = preprocess_img(img)
    source_desc.append(net.submit(img_arr).squeeze())
    
for img_name in target_imgs.split('|'):
    img = Image.open(os.path.join(imgs_path,img_name))
    img_arr = preprocess_img(img)
    target_desc.append(net.submit(img_arr).squeeze())


dist = lambda x, y: np.round(np.sqrt(((x - y) ** 2).sum(axis=0)),4)

print('Dist between S0_Im0 and S0_Imi:',list(map(dist,5*[source_desc[0]],source_desc)))
print('Dist between S0_Im0 and T0_Imi:',list(map(dist,5*[source_desc[0]],target_desc)))
def img_to_crop(img):
    preprocessing = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.Resize(112),
                    ])
    return preprocessing(img)


def crop_to_tensor(img):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                    ])
    img_arr = preprocessing(img).unsqueeze(0).numpy()
    return img_arr

from skimage.io import imsave, imread


img_name=source_imgs.split('|')[0]
img = Image.open(os.path.join(imgs_path,img_name))
img_crop = img_to_crop(img)

img_crop.save('tmp.png')
img_crop.save('tmp.jpg')
img_crop_jpg=Image.open('tmp.jpg')
img_crop_png=Image.open('tmp.png')
ssim(np.array(img_crop_jpg), np.array(img_crop_png), multichannel=True)

img_crop_jpg=Image.open('tmp.jpg')
img_crop_png=Image.open('tmp.png')
ssim(np.array(img_crop_jpg), np.array(img_crop_png), multichannel=True)





print('ended')
