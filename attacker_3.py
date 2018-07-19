import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from gpu_model import start_gpu_thread

start_gpu_thread('../data/pairs_list3.csv','./logs/3')


