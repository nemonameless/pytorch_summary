#coding:utf-8
import time
import torch
from torch.autograd import Variable
def runtime(model, inputsize=(3,224,224), iter = 10, device='cpu'):
    model.eval()
    print("Testing runtime using", torch.cuda.get_device_name(0))
    if device == "gpu" and torch.cuda.is_available():
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad = True).cuda()
    else:
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad = True)
    sum_time = 0
    for i in range(iter):
        torch.cuda.synchronize()
        start_time = time.time()
        out = model(input)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        sum_time += time_taken
    print("Run-Time: %.4f ms"% (sum_time/float(iter)*1000))
