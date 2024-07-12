# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:21:57 2022

@author: GengxiangCHEN
"""

import torch
import torch.nn.functional as F
import numpy as np
from lapy import TriaMesh,Solver,TetMesh
from timeit import default_timer
import scipy.io as sio
import time
import pandas as pd
from utilities3 import RangeNormalizer,count_params,LpLoss,UnitGaussianNormalizer, GaussianNormalizer ,GetDLO ,RangeNormalizer
from utilsLNO import MeshNO
import os
from Adam import Adam
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#torch.manual_seed(0)
#np.random.seed(0)

def main(args):  

    # print("\n=============================")
    # print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    # if torch.cuda.is_available():
    #     print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    # print("=============================\n")
    # print('M',args.modes,' W',args.width,' B',args.batch_size,' LR', args.lr,' BS-', args.basis)

    ################################################################
    # configs
    ################################################################
    
    PATH = args.data_dir
 
    ntrain = args.num_train
    ntest = args.num_test
    
    batch_size = args.batch_size
    learning_rate = args.lr
    
    epochs = args.epochs
    
    modes = args.modes
    width = args.width
    
    step_size = 200
    gamma = 0.5
    
    #s = args.size_of_nodes

    ################################################################
    # reading data and normalization
    ################################################################   
    Data = sio.loadmat(PATH)
    
    
    ntrain = args.num_train
    ntest  = args.num_test
    dataX = torch.tensor(Data['Tair_time'].astype(np.float32))
    # dataX1 = dataX[:, 0, :]
    # dataX2 = dataX[:, 1, :]
    # dataX = torch.cat((dataX1, dataX2), 1)

    x_train = dataX[0:ntrain]
    x_test  = dataX[-ntest:]
    
    y_train = torch.Tensor(Data['D_field'][0:ntrain:,:] )
    y_test  = torch.Tensor(Data['D_field'][-ntest:,:] )

    n_t = y_test.shape[1]

    # x_train = x_train.reshape(ntrain, -1, 1)
    # x_test = x_test.reshape(ntest, -1, 1)
    
    norm_x  = RangeNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)

    
    norm_y  = RangeNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
    
    print(ntrain,ntest)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    

    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)

    ###basis 1D
    DLO = GetDLO(modes)
    L_matrix = DLO.Get1dLaplace(x_train.shape[1])
    # L_right = DLO.Get1dLaplace_right(Nt, L_matrix)
    # L_left = DLO.Get1dLaplace_left(Nt, L_matrix)
    # L_mid = L_left + L_right

    eigenvalue, E = DLO.GetE(L_matrix)  # 此时截断到k个特征向量
    E_inverse = E.T
    E = torch.Tensor(E).cuda()
    E_inverse = torch.Tensor(E_inverse).cuda()

    k = 128
    s = n_t
    # Points = np.vstack((Data['MeshNodes'].T, np.zeros(s).reshape(1, -1)))#2XNODES
    # mesh = TriaMesh(Points.T, Data['MeshElements'] - 1)#3XELEMENTS
    points = Data['nodes']
    faces = Data['elements'] - 1
    faces = faces.astype(int)
    # tetra = TetMesh(points, faces)
    mesh = TetMesh(points, faces)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)
    lbo_basis = torch.Tensor(LBO_MATRIX[:, :modes])

    MATRIX = torch.Tensor(lbo_basis).cuda()
    INVERSE = (MATRIX.T @ MATRIX).inverse() @ MATRIX.T
    
    print('BASE_MATRIX:', MATRIX.shape, 'BASE_INVERSE:', INVERSE.shape)
    

    
    model = MeshNO(modes, width, MATRIX, INVERSE,E ,E_inverse).cuda()
    
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    # ET_list = np.zeros((epochs))
    
    for ep in range(epochs):
        model.train()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x)
    
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   

            # loss_max_train = (abs(out.view(batch_size, -1)- y.view(batch_size, -1))).max(axis=1).values.mean()
            # loss_max_train = loss_max_train * torch.std(y_dataIn) + torch.mean(y_dataIn)
    
            optimizer.step()
            train_mse += mse.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = model(x)
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
                
                test_l2 += myloss(out_real, y_real).item()                
                # loss_max_test= (abs(out.view(batch_size, -1)- y.view(batch_size, -1))).max(axis=1).values.mean()
                # loss_max_test = loss_max_test * torch.std(y_dataIn) + torch.mean(y_dataIn)
    
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        
        # ET_list[ep] = loss_max_test
        # print("Idx: %u L_tr: %2.3f L_te: %2.3f Emax_tr: %2.3f Emax_te: %2.3f"  % (ep, train_l2, test_l2, loss_max_train, loss_max_test))
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 10 == 0:
            print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
        time_step = time.perf_counter()
          
    # print("\n=============================")
    print("Training done...")
    # print("=============================\n")
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    pre_train = torch.zeros(y_train.shape)
    y_train   = torch.zeros(y_train.shape)
    # x_test    = torch.zeros(x_train.shape[0:2])
    
    index = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            
            pre_train[index,:] = out_real
            y_train[index,:]   = y_real
            
            index = index + 1
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test   = torch.zeros(y_test.shape)
    x_test   = torch.zeros(x_test.shape)
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.cpu())
            x_real   = norm_x.decode(x.cpu())
            
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index,:] = x_real
            
            index = index + 1
            
    # ================ Save Data ====================
    sava_path = "../Results/04.NORM/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)

    dataframe = pd.DataFrame({'Test_loss': [test_l2],
                              'Train_loss': [train_l2],
                              'num_paras': [count_params(model)],
                              'train_time': [time_step_end - time_start]})

    dataframe.to_csv(sava_path + 'log.csv', index=False, sep=',')

    dataframe2 = pd.DataFrame({'width': [width],
                               'modes': [modes],
                               'epochs': [epochs],
                               'lr': [learning_rate],
                               'ntarin': [ntrain],
                               'ntest': [ntest],
                               'batch_size': [batch_size]})
    dataframe2.to_csv("../Results/04.NORM/" + args.CaseName + 'paraments.csv', index=False, sep=',')

    model_output = model.state_dict()
    loss_dict = {'train_error': train_error,
                 'test_error': test_error}

    pred_dict = {'pre_test': pre_test.cpu().detach().numpy(),
                 'pre_train': pre_train.cpu().detach().numpy(),
                 # 'x_test'   : x_test.cpu().detach().numpy(),
                 # 'x_train'  : x_train.cpu().detach().numpy(),
                 'y_test': y_test.cpu().detach().numpy(),
                 'y_train': y_train.cpu().detach().numpy(),
                 }

    torch.save(model_output, "../Results/04.NORM/" + args.CaseName + "/" + '_MeshNO_net_params.pkl')
    sio.savemat(sava_path + 'MeshNO_loss.mat', mdict=loss_dict)
    sio.savemat(sava_path + 'MeshNO_pre.mat', mdict=pred_dict)

    test_l2 = (myloss(y_test, pre_test).item())/ntest
    print('\nTesting error: %.3e'%(test_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))
    # print('M',args.modes,' W',args.width,' B',args.batch_size,' LR', args.lr,' BS-', args.basis)


if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    # Case1:    
    for i in range(5):
        
        print('====================================')
        print('NO '+str(i)+' train......')
        print('====================================')

        for args in [
                        { 'modes'   : 128,
                          'width'   : 64,
                           # 'size_of_nodes' : 2282,
                          'batch_size': 10, 
                          'epochs'    : 1000,
                          'data_dir'  : '../Data/Qianyuan_T2_2.mat',
                          'num_train' : 500,
                          'num_test'  : 100,
                          'CaseName'  : 'qianyuan2_e3_1000_07/' + str(i), #此处保留测试记录用
                          'basis'     : 'LBO',
                          'lr'        : 0.001},
                    ]:
            
            args = objectview(args)
                
        main(args)




    