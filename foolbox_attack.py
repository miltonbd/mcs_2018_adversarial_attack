import foolbox
import torch
import torchvision
import numpy as np
from classification.models.pytorch import vgg


torchmodel = vgg.vgg19_bn()
torchmodel.eval()
model = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 255), num_classes=1000)


# get source image and label
image, label = foolbox.utils.imagenet_example()

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(torchmodel)
img=torch.from_numpy(np.flip(image[:, :, ::-1],axis=0).copy())
img=img.unsqueeze(0)
print(img.shape)
adversarial = attack(img, label)

import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()