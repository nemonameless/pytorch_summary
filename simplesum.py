#coding:utf-8
from utils.parm import print_model_parm_flops
from utils.parm import print_model_parm_nums
import torchvision.models as models

model = models.resnet50()

print_model_parm_nums(model)
print_model_parm_flops(model,inputsize= (3,224,224))
