#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: model_tools.py
@time: 19-5-31 上午10:48
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from graphviz import Digraph
from tool.flops import count_ops

class ModelTools(object):
    def __init__(self, input, model):
        self.model = model
        self.input = input

    @staticmethod
    def parameters_total(model):
        total = sum(param.numel() for param in model.parameters())
        return total / 1e6

    def print_parameters_total(self):
        total = self.parameters_total(self.model)
        print('Number of params: %.2f M' % (total))


    @staticmethod
    def model_flops(input_tensor, model):
        estimated = count_ops(model, input_tensor, print_readable=False)
        return estimated / 1e9

    def print_model_flops(self):
        flops = self.model_flops(self.input,self.model)
        print('Number of flops: %.2fG' % (flops))


    #TODO: what fuck
    @staticmethod
    def model_forward(input_tensor, model):
        select_layer = model.layer1[0].conv1
        grads = {}

        def save_grad(name):
            def hook(self, input, output):
                grads[name] = input
            return hook

        select_layer.register_forward_hook(save_grad('select_layer'))
        input = input_tensor.requires_grad_(True)
        model(input)
        return grads

    def print_model_forward(self):
        grads = self.model_forward(self.input, self.model)
        print('model forward: \n',grads)


    @classmethod
    def summarize(cls, model, params_switch=True, weights_switch=True):
        """
            Summarizes torch model by showing trainable parameters and weights.
        """
        from torch.nn.modules.module import _addindent
        tmpstr = model.__class__.__name__ + '(\n'
        for key, module in model._modules.items():
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = cls.summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            tmpstr += '  ('+key + '): ' + modstr
            if params_switch:
                params = sum([np.prod(p.size()) for p in module.parameters()])
                tmpstr += ',  parameters={}'.format(params)
            if weights_switch:
                weights = tuple([tuple(p.size()) for p in module.parameters()])
                tmpstr += ',  weights={}'.format(weights)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        return tmpstr

    def print_summarize(self, params_switch=True, weights_switch=True):
        tmpstr = self.summarize(self.model, params_switch, weights_switch)
        print(tmpstr)

    @staticmethod
    def __get_names_dict(model):
        """Recursive walk to get names including path"""
        names = {}
        def _get_names(module, parent_name=''):
            for key, module in module.named_children():
                name = parent_name + '.' + key if parent_name else key
                names[name] = module
                if isinstance(module, torch.nn.Module):
                    _get_names(module, parent_name=name)
        _get_names(model)
        return names


    @classmethod
    def keras_summarize_like(cls,input_tensor, model, weights=False, input_shape=True, trainable=False):
        """
        Summarizes torch model by showing trainable parameters and weights.

        author: wassname
        url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
        license: MIT

        Modified from:
        - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
        - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/

        Usage:
            import torchvision.models as models
            model = models.alexnet()
            df = torch_summarize_df(input_size=(3, 224,224), model=model)
            print(df)

            #              name class_name        input_shape       output_shape  nb_params
            # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296#(3*11*11+1)*64
            # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
            # ...
        """
        names = cls.__get_names_dict(model)
        summary = OrderedDict()
        hooks = []

        def register_hook(module):
            def hook(module, input, output):
                name = ''
                for key, item in names.items():
                    if item == module:
                        name = key
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = module_idx +1
                summary[m_key] = OrderedDict()
                summary[m_key]['name'] = name
                summary[m_key]['class_name'] = class_name

                if input_shape:
                    summary[m_key]['input_shape']=(-1, )+tuple(input[0].size())[1:]
                    summary[m_key]['output_shape']=(-1,)+tuple(output.size())[1:]
                if weights:
                    summary[m_key]['weights']=list(
                        [tuple(p.size()) for p in module.parameters()])

                if trainable:
                    params_trainable = sum(
                        [torch.LongTensor(list(p.size())).prod()
                             for p in module.parameters() if p.requires_grad])
                    summary[m_key]['nb_trainable'] = params_trainable

                params = sum([torch.LongTensor(list(p.size())).prod()
                              for p in module.parameters()])
                summary[m_key]['nb_params'] = params

            if not isinstance(module, torch.nn.Sequential) and \
                not isinstance(module, torch.nn.ModuleList) and \
                not (module == model):
                hooks.append(module.register_forward_hook(hook))

        input_tensor.requires_grad_(True)

        if next(model.parameters()).is_cuda:
            input_tensor.to('cuda')

        model.apply(register_hook)
        model(input_tensor)

        for h in hooks:
            h.remove()

        df_summary = pd.DataFrame.from_dict(summary, orient='index')

        return df_summary

    def print_keras_summary_like(self,
                                 weights=True, input_shape=True, trainable=True):
        dataFrame = self.keras_summarize_like(self.input,self.model,weights, input_shape, trainable)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 800)
        pd.set_option('display.max_rows', None)
        print(dataFrame)

    @staticmethod
    def visualization_weights(weights, ch=0, file_name=None, all_kernels=False, nrow=8, padding=2):
        """
        :param weight:
        :param ch: channel for visualization
        :param file_name: if it is None, don't save the weight
        :param all_kernels: all kernels for visualization
        :param nrow:
        :param padding:
        :return:
        """
        from torchvision import utils

        n,c,h,w = weights.shape
        if all_kernels:
            weights = weights.view(n*c, -1, w, h)
        elif c!=3:
            weights = weights[:, ch, :, :].unsqueeze(dim=1)

        grad = utils.make_grid(weights, nrow=nrow,padding=padding,normalize=True)
        if file_name:
            utils.save_image(weights,file_name, nrow=nrow, normalize=True, padding=padding)
        plt.imshow(grad.numpy().transpose((1,2,0)))#CHW to HWC

    @classmethod
    def run_visualization_weights(cls):
        model = torchvision.models.resnet18(pretrained=True)
        mm = model.double()
        body_model = [i for i in mm.children()]
        layer1 = body_model[0]
        tensor = layer1.weight.data.clone()
        cls.visualization_weights(tensor, file_name='weight.png')


    @staticmethod
    def make_dot(var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph.

        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function

        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert all(torch.is_tensor(p) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    # note: this used to show .saved_tensors in pytorch0.2, but stopped
                    # working as it was moved to ATen and Variable-Tensor merged
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                elif var in output_nodes:
                    dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        # handle multiple outputs
        if isinstance(var, tuple):
            for v in var:
                add_nodes(v.grad_fn)
        else:
            add_nodes(var.grad_fn)

        def resize_graph(dot, size_per_element=0.15, min_size=12):
            """Resize the graph according to how much content it contains.

            Modify the graph in place.
            """
            # Get the approximate number of nodes and edges
            num_rows = len(dot.body)
            content_size = num_rows * size_per_element
            size = max(min_size, content_size)
            size_str = str(size) + "," + str(size)
            dot.graph_attr.update(size=size_str)

        resize_graph(dot)

        return dot

    @classmethod
    def visualization_compute_graph_deprecated(cls):
        torch.manual_seed(1)
        input = torch.randn(1,3,224,224,requires_grad=True)
        model = torchvision.models.resnet18(pretrained=False)
        y = model(input)
        g = cls.make_dot(y, params=model.state_dict())
        g.render('resnet18_compute_graph/resnet18', format='png')

    @staticmethod
    def visualization_compute_graph():
        import hiddenlayer as hl
        # VGG16 with BatchNorm
        model = torchvision.models.resnet18()

        # Build HiddenLayer graph
        # Jupyter Notebook renders it automatically
        g = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
        g.save("resnet18_compute_graph/resnet18",format='png')


    def print_info(self):
        self.print_parameters_total()
        self.print_model_flops()
        self.print_model_forward()
        # self.print_summarize()
        self.print_keras_summary_like()

def main():
    net = torchvision.models.resnet18()
    input = torch.rand(1, 3, 224,224)
    tools = ModelTools(input, net)
    tools.print_parameters_total()
    tools.print_model_flops()
    #tools.print_model_forward()
    # tools.print_summarize()
    tools.print_keras_summary_like()
    # tools.run_visualization_weights()
    # plt.pause(1)
    #tools.visualization_compute_graph()

if __name__ == "__main__":
    import fire
    fire.Fire(main)