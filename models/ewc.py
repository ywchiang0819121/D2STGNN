import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import numpy as np
import logging
import pdb
from utils.train import data_reshaper

from torch_geometric.data import Data


class EWC(nn.Module):

    def __init__(self, model, args, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type
        self.args = args

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loader, lossfunc, device):
        emb_names = ['module.node_emb_u', 'module.node_emb_d']
        _buff_param_names = [param[0].replace('.', '__') 
                            for param in self.model.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        for itera, (x, y) in enumerate(loader.get_iterator()):
            if self.args.cur_year > self.args.begin_year and self.args.strategy == 'incremental':
                x = x[:, :, self.args.subgraph, :] 
                y = y[:, :, self.args.subgraph, :]
            x = data_reshaper(x, device=device)
            y = data_reshaper(y, device=device)
            pred = self.model.forward(x)
            log_likelihood = lossfunc(y, pred)
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                if grad == None or name in ['module__node_emb_u', 'module__node_emb_d']:
                    est_fisher_info[name] = None
                else:
                    est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])


    def register_ewc_params(self, loader, lossfunc, device):
        self._update_fisher_params(loader, lossfunc, device)
        self._update_mean_params()


    def compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            # print(param_name, param.size(),estimated_mean.size(), estimated_fisher.size())
            if estimated_fisher == None:
                losses.append(0)
            elif self.ewc_type == 'l2':
                losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses)
    
    def forward(self, data): 
        return self.model(data)

