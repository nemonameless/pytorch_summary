# -*- coding: utf-8 -*-

import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.module_madd import compute_module_madd


class CModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = Variable(torch.rand(1, *self._input_size))  # add module duration time
        self._model.eval()
        self._model(x)

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        module.register_buffer('input_shape', torch.zeros(3).int())
        module.register_buffer('output_shape', torch.zeros(3).int())
        module.register_buffer('parameter_quantity', torch.zeros(1).int())
        module.register_buffer('inference_memory', torch.zeros(1).long())
        module.register_buffer('MAdd', torch.zeros(1).long())
        module.register_buffer('duration', torch.zeros(1).float())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            module.input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.long))

            inference_memory = 1
            for s in output.size()[1:]:
                inference_memory *= s
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            madd = compute_module_madd(module, input, output)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            return output

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    def _retrieve_leaf_modules(self, model, prefix=''):
        leaf_modules = []
        for name, module in model._modules.items():
            name = prefix + ('' if prefix == '' else '.') + name
            if len(list(module.children())) > 0:
                leaf_modules += self._retrieve_leaf_modules(module, name)
            else:
                # add operation type name
                name = name + '_' + str(module)[0:str(module).find('(')]
                leaf_modules.append((name, module))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))


def main():
    pass


if __name__ == "__main__":
    main()
