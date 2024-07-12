# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:59:41 2023

@author: Mxm
"""
#将血流数据处理成志良的研究场景
import scipy.io as sio

Data = sio.loadmat('Traindata_2.mat')

# In[]

import numpy as np
BC_time = Data['BC_time']

velocity_x = Data['velocity_x']
velocity_y = Data['velocity_y']
velocity_z = Data['velocity_z']

nodes = Data['nodes']
elements = Data['elements']

Input = np.zeros((BC_time.shape[0],BC_time.shape[1],BC_time.shape[2]))

Output_x = np.zeros((velocity_x.shape[0],velocity_x.shape[1]))
Output_y = np.zeros((velocity_x.shape[0],velocity_x.shape[1]))
Output_z = np.zeros((velocity_x.shape[0],velocity_x.shape[1]))

for i in range(BC_time.shape[0]):
    
    Input[i] = BC_time[i]
    
    max_value_v = np.max(BC_time[i,:,0])
    index1 = np.argwhere(BC_time[i,:,0] == max_value_v)
    print(index1)
    
    # max_value_p = np.max(BC_time[i,:,1])
    # index2 = np.argwhere(BC_time[i,:,1] == max_value_p)
    # print(index2)
    # print('-----------')#二者最大值的索引应该相同
    
    Output_x[i] = velocity_x[i,:,index1]
    Output_y[i] = velocity_y[i,:,index1]
    Output_z[i] = velocity_z[i,:,index1]
    
sio.savemat('BloodFlow.mat', 
             { 'nodes':  nodes,
               'elements': elements,
               'BC_time':  Input,
               'velocity_x': Output_x,
               'velocity_y': Output_y,
               'velocity_z': Output_z})
    
    
    
    
    
    
    
    
    
    