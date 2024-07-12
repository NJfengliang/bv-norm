# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:38:35 2022

@author: GengxiangCHEN
"""

from importlib import reload
import lapy
import pyvista as pv
from lapy import Solver, Plot, TriaIO, TetIO, FuncIO, TetMesh, TriaMesh
reload(Plot)
import scipy.io as sio
import plotly.io as pio
pio.renderers.default='browser'
import numpy as np
mesh   = sio.loadmat('Part_Mesh.mat')
points = mesh['points']
faces  = mesh['elements'] - 1
faces  = faces.astype(int)
# faces = mesh.faces.reshape(-1, 4)

k = 128
tetra = TetMesh(points,faces)
fem   = Solver(tetra)
evals, evecs = fem.eigs(k = k)
evDict = dict()
evDict['Refine'] = 0
evDict['Degree'] = 1
evDict['Dimension'] = 2
evDict['Elements'] = len(tetra.t)
evDict['DoF'] = len(tetra.v)
evDict['NumEW'] = k
evDict['Eigenvalues'] = evals
evDict['Eigenvectors'] = evecs
evDict['Points'] = points
evDict['Faces'] = faces

print(evecs.shape)

sio.savemat('lbe_ev_output.mat', evDict)  


# import pyvista as pv
# mesh.plot()