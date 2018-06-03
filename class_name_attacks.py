import MCS2018
from PIL import Image
from torchvision import transforms
import glob

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

net1 = MCS2018.Predictor(1)
preprocessing = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Scale(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

for img_name in glob.glob("/media/milton/ssd1/research/competitions/data/imgs/*"):
    img = Image.open(img_name)
    img_arr = preprocessing(img).unsqueeze(0).numpy()
    # if idx%2==0:
    #     res = net.submit(img_arr).squeeze()
    # else:
    res = net1.submit(img_arr)
    print(res.shape)
    break