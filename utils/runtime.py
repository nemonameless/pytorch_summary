#coding:utf-8
import time
import torch
from torch.autograd import Variable
def runtime(model, inputsize=(3,224,224), iter = 10, device=-1):
    model.eval()
    
    if device>=0 and torch.cuda.is_available():
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad = True).cuda(device)
        print("Testing runtime using", torch.cuda.get_device_name(device))
    else:
        input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad = True)
        print("Testing runtime using cpu")
    sum_time = 0
    for i in range(iter):
        start_time = time.time()
        out = model(input)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        sum_time += time_taken
    print("Run-Time: %.4f ms"% (sum_time/float(iter)*1000))

