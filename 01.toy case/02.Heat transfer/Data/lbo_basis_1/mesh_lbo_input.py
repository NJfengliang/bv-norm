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
mesh = pv.read('0620_mesh_input.stl')
points = mesh.points
faces = mesh.faces.reshape(-1, 4)
faces = faces[:,1:4]

# Points = np.vstack((data['MeshNodes'], np.zeros(s).reshape(1,-1)))

# faces = mesh.faces.reshape(-1, 4)

k = 128
tetra = TriaMesh(points,faces)
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

# print(evecs.shape)

sio.savemat('lbe_ev_input.mat', evDict)  


# import pyvista as pv
# mesh.plot()