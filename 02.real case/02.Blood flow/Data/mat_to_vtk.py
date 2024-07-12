# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pyvista as pv
import scipy.io as sio
import pyvista as pv
import pandas as pd
import vtk
from pyvista import CellType
import matplotlib.pyplot as plt


d_ = sio.loadmat('BloodFlow.mat')


t = 1

print('数据导入完成')

# v_input        = LNO_V['x_test']
# pressure = LNO['x_test'][:,:,:]

velocity_x = d_['velocity_x']
velocity_y = d_['velocity_y']
velocity_z = d_['velocity_z']

# velocity_x_pre = x_['pre_test']
# velocity_y_pre = y_['pre_test']
# velocity_z_pre = z_['pre_test']


# velocity_x_DON = DON_Vx['pre_test'].transpose(0, 2, 1)
# velocity_y_DON = DON_Vy['pre_test'].transpose(0, 2, 1)
# velocity_z_DON = DON_Vz['pre_test'].transpose(0, 2, 1)

# velocity_x_PDN = PDN['pre_test'][:,:,:,0].transpose(0, 2, 1)
# velocity_y_PDN = PDN['pre_test'][:,:,:,1].transpose(0, 2, 1)
# velocity_z_PDN = PDN['pre_test'][:,:,:,2].transpose(0, 2, 1)

# velocity_DeepOnet = np.sqrt(velocity_x_DON**2 + velocity_y_DON**2 + velocity_z_DON**2)
# velocity_POD_DeepOnet = np.sqrt(velocity_x_PDN**2 + velocity_y_PDN**2 + velocity_z_PDN**2)

nodes = pd.read_csv('coordinates.csv',header=None).values
elements = pd.read_csv('elements.csv',header=None).values-1
print('数据导入完成')



# pressure_DeepOnet = DON_P['pre_test'].transpose(0, 2, 1)
# pressure_POD_DeepOnet = PDN_P['pre_test'].transpose(0, 2, 1)



# v_input        = LNO_V['x_test']

# # pressure       = LNO_P['y_test']

# velocity_x     = LNO_Vx['y_test']
# velocity_y     = LNO_Vy['y_test']
# velocity_z     = LNO_Vz['y_test']

# velocity_x_LNO = LNO_Vx['pre_test']
# velocity_y_LNO = LNO_Vy['pre_test']
# velocity_z_LNO = LNO_Vz['pre_test']

# velocity_x_DON = DON_Vx['pre_test'].transpose(0, 2, 1)
# velocity_y_DON = DON_Vy['pre_test'].transpose(0, 2, 1)
# velocity_z_DON = DON_Vz['pre_test'].transpose(0, 2, 1)

# velocity_x_PDN = PDN_Vx['pre_test'].transpose(0, 2, 1)
# velocity_y_PDN = PDN_Vy['pre_test'].transpose(0, 2, 1)
# velocity_z_PDN = PDN_Vz['pre_test'].transpose(0, 2, 1)

# mesh.plot(show_edges=True, opacity=0.25)
# index = 3

celltypes = np.full(elements.shape[0], fill_value=CellType.TETRA, dtype=np.uint8)
cells = np.hstack((np.full((elements.shape[0], 1), 4), elements))

mesh1 = pv.UnstructuredGrid(cells, celltypes, nodes)
# mesh2 = pv.UnstructuredGrid(cells, celltypes, nodes)

case_ID = 11

# for i in range((velocity_x.shape[-1])):
    
            
    # mesh1.point_data['pressure'] = pressure[case_ID,:,i]
    # mesh1.point_data['v_input'] = v_input[case_ID,i,0]
    # c = velocity_x[case_ID,i]
    
mesh1.point_data['v_vector'] = np.vstack((
    velocity_x[case_ID,:],
    velocity_y[case_ID,:],
    velocity_z[case_ID,:],
    )).T

# mesh1.point_data['v_vector_pre'] = np.vstack((
#     velocity_x_pre[case_ID,:],
#     velocity_y_pre[case_ID,:],
#     velocity_z_pre[case_ID,:],
#     )).T
#
# mesh1.point_data['v_error'] = np.vstack((
#     abs(velocity_x_pre[case_ID,:]-velocity_x[case_ID,:]),
#     abs(velocity_y_pre[case_ID,:]-velocity_y[case_ID,:]),
#     abs(velocity_z_pre[case_ID,:]-velocity_z[case_ID,:]),
#     )).T



mesh1.save('TRUTH'+'.vtk')



