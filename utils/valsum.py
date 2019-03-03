#coding:utf-8
import torch
from torchsummary import summary

def valsum(model, input_size, device = 'cpu'):
    summary(model, tuple(input_size))

# https://github.com/sksq96/pytorch-summary
# pip3 install torchsummary
