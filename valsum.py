#coding:utf-8
import torch
from torchvision import models
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16().to(device)

summary(model, (3, 224, 224))

# https://github.com/sksq96/pytorch-summary
# pip3 install torchsummary
