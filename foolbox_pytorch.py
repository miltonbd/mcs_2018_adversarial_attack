import foolbox
import numpy as np
from student_net_learning import model_loader

torchmodel=model_loader.get_model_net("")
torchmodel.cuda()
torchmodel.eval()
fmodel = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 255), num_classes=512)
# get source image and label
image1, label = foolbox.utils.imagenet_example(shape=(112,112))

image=image1.transpose((2, 0, 1)).copy()

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)

import matplotlib.pyplot as plt

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image1 / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
adversarial1= np.rot90(adversarial.transpose((2, 1, 0)).copy(),3)
plt.imshow(adversarial1 / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
difference = difference.transpose((2,1,0))
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()