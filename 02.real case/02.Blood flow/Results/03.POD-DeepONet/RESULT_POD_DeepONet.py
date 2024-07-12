# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:08:23 2023

@author: Chenl
"""

import numpy as np
import scipy.io as sio
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        # print(x.shape, y.shape)
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
def loss_func(y_true, y_pre):
    
    print(y_true.shape)
    # MAPE = np.mean(np.abs((y_true - y_pre) / y_true))
    MAX  = np.max(np.abs(y_true - y_pre))
    MMAX = np.mean(np.max(np.abs(y_true - y_pre), axis = 1))
    
    myloss = LpLoss(size_average=False)
    lploss = myloss(torch.Tensor(y_pre), torch.Tensor(y_true)).item()/y_true.shape[0]
    
    result = np.array([MAX, MMAX, lploss*100])

    return result

if __name__ == '__main__':
    
    n = 5
    
    Results = np.zeros((n,7))
    for i in range (n):
        
        data = sio.loadmat(str(i) + '/POD_DeepONet_pre.mat')
        y_train = data['y_train']
        y_test = data['y_test']
        pre_train = data['pre_train']
        pre_test = data['pre_test' ]
        
        Results[i, 0:3] = loss_func(y_test, pre_test)
        Results[i, 4:7] = loss_func(y_train, pre_train)
    
    std  = np.std (Results, axis=0).reshape((1,-1))
    mean = np.mean(Results, axis=0).reshape((1,-1))
    
    sio.savemat('RESULTS.mat', {'result': Results,
                                'mean'  : mean,
                                'std'   : std})
    
    print('TEST L2 Error: ', mean[0,2])
    
    
    
    
    
    
    
    
    
    
    
    
    