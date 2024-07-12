# -*- coding:utf-8 -*-
# @Time ： 2024/2/1 16:15
# @Author ： Fengliang
# @File ： plot_basis.py
# @Software： PyCharm

# -*- coding:utf-8 -*-
# @Time ： 2024/1/31 11:14
# @Author ： Fengliang
# @File ： lbo.py
# @Software： PyCharm


import numpy as np
import matplotlib.pyplot as plt
import random

import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 设置公式字体 STIX
plt.rc('font', family='arial', size=12)

basis_lbo_bc = sio.loadmat('Data/lbo_basis_3/lbe_ev_input.mat')['Eigenvectors'].astype(np.float32)
basis_lbo_u  = sio.loadmat('Data/lbo_basis_3/lbe_ev_output.mat')['Eigenvectors'].astype(np.float32)
data = sio.loadmat('Data/data_heat_3.mat')
bc = data["input"].astype(np.float32)
u = data[ "output"].astype(np.float32)

def pod(y):
    n = len(y)
    y = np.array(y)
    y_mean = np.mean(y, axis=0)
    #y = y - y_mean

    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    return v

# POD
#---------------------------------------------------------------------------------
W_in = pod(bc).copy()
W_out = pod(u).copy()
w_bc = W_in[:128].T
w_u = W_out[:128].T
# print("hellow")
INDE = 6
# basis_inv
#---------------------------------------------------------------------------------
L_bc_INV = np.linalg.pinv(basis_lbo_bc.T @ basis_lbo_bc) @ basis_lbo_bc.T
L_u_INV = np.linalg.pinv(basis_lbo_u.T @ basis_lbo_u) @ basis_lbo_u.T
P_bc_INV = np.linalg.pinv(w_bc.T @ w_bc) @ w_bc.T
P_u_INV = np.linalg.pinv(w_u.T @ w_u) @ w_u.T

# coefficient
#---------------------------------------------------------------------------------
cL_bc_INV = L_bc_INV @ bc.T
cL_u_INV = L_u_INV @ u.T
cP_bc_INV = P_bc_INV @ bc.T
cP_u_INV = P_u_INV @ u.T

# Plot
#---------------------------------------------------------------------------------
x_inde = np.linspace(1,128,128)
fig, axs = plt.subplots(figsize=(10,8))
plt.subplots_adjust(left=0.10, right=0.95,bottom=0.06,top=0.9,wspace=0.2)

# segment = 'darcy'
grid = plt.GridSpec(2,2, wspace=0.2, hspace=0.3)
plt.subplot(grid[0,0])
plt.bar(x_inde, cL_bc_INV[:,INDE], color='#8FC2C7', alpha=0.7)
plt.subplot(grid[0,1])
plt.bar(x_inde, cL_u_INV[:,INDE], color='#BD514A', alpha=0.7)
plt.subplot(grid[1,0])
plt.bar(x_inde, cP_bc_INV[:,INDE], color='#8FC2C7', alpha=0.7)
plt.subplot(grid[1,1])
plt.bar(x_inde, cP_u_INV[:,INDE], color='#BD514A', alpha=0.7)
plt.show()