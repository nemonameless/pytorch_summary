import argparse
import torch
import os
from utils.simplesum import simplesum
from utils.complexsum import complexsum
from utils.valsum import valsum
parser = argparse.ArgumentParser(description='PyTorch Summary')

parser.add_argument('--mod', default='simple', type=str, 
                    help='simple complex val')
parser.add_argument('--gpu', default='-1',type=str,
                    help='GPU: ID or CPU: -1')
parser.add_argument('--size', default="3,224,224", type=str, 
                    help='Size of input image (C,H,W)')


args = parser.parse_args()
###### add your model ######
from torchvision import models
model = models.resnet50()
############################
if int(args.gpu) >=0:
  if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Using GPU: ",args.gpu)
    device = torch.device('cuda')
    device_mode = 'gpu'
  else:
    print("CUDA is not available.")
    device = torch.device('cpu')
    device_mode = 'cpu'
else:    
  print("Using CPU")
  device = torch.device('cpu')
  device_mode = 'cpu'

model = model.to(device)
inputsize = args.size.split(',')
inputsize = [int(x) for x in inputsize]
print("Input size: ", inputsize)
if args.mod =='simple':
    print("Using SIMPLE summary mode:")
    simplesum(model, inputsize, device_mode)
elif args.mod =='complex':
    print("Using COMPLEX summary mode:")
    complexsum(model, inputsize, device_mode)
elif args.mod =='val':
    print("Using VALIDATION summary mode: (Only support GPU mode.)")
    valsum(model, inputsize, device_mode)
else:
    print("Only support simple|complex|val modes.")

