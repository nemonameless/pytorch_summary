import argparse
import torch
import os
from utils.simplesum import simplesum
from utils.complexsum import complexsum
from utils.valsum import valsum
from utils.runtime import runtime
parser = argparse.ArgumentParser(description='PyTorch Summary')

parser.add_argument('--mod', default='simple', type=str, 
                    help='simple complex val')
parser.add_argument('--gpu', default='-1',type=int,
                    help='GPU: ID or CPU: -1')
parser.add_argument('--size', default="3,224,224", type=str, 
                    help='Size of input image (C,H,W)')
parser.add_argument('--runtime', default='-1',type=int,
                    help='-1: not enable runtime test. runtime>=1 average iters for runtime test.')


args = parser.parse_args()
###### add your model ######
from torchvision import models
model = models.resnet50()
############################
if args.gpu >=0:
  if torch.cuda.is_available():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Using GPU: ",args.gpu)
    device = torch.device('cuda', args.gpu)
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
    model.eval()
    simplesum(model, inputsize, device = args.gpu)
elif args.mod =='complex':
    print("Using COMPLEX summary mode:")
    complexsum(model, inputsize, device = args.gpu)
elif args.mod =='val':
    print("Using VALIDATION summary mode: (Only support GPU mode.)")
    valsum(model, inputsize, device = args.gpu)
else:
    print("Only support simple|complex|val modes.")

if args.runtime>0:
    runtime(model, inputsize, iter = args.runtime, device = args.gpu)
