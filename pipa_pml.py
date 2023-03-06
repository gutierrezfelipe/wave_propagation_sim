#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:08:26 2022

@author: danielpipa
"""

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from scipy.ndimage import correlate
import math

dtype = 'float32'  # float32 is the fastest

Dx = .02e-3
Dz = Dx
Dt = 1e-9

# PML
R, Np = 1e-5, 10
R, Np = 1e-5, 20
# R, Np = 1e-2, 30
Lp = Np * Dx

Lx = 5e-3 + Lp  # Block width
Lz = 5e-3 + Lp  # Block height
Lt = 3e-6  # Simulation time

Nx = round(Lx / Dx)
Nz = round(Lz / Dz)
Nt = round(Lt / Dt)

c = 5490
c = 1480

C2 = c ** 2 * Dt ** 2 / (Dx ** 2)

sfmt = pg.QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
pg.QtGui.QSurfaceFormat.setDefaultFormat(sfmt)
app = pg.QtGui.QApplication([])
# app = pg.QtWidgets.QApplication([])
riw = RawImageGLWidget(scaled=True)
riw.show()

# Source signal
t = np.linspace(0, Lt - Dt, Nt, dtype=dtype)
f0 = 5e6  # Transducer central frequency
bwp = .9  # Bandwidth in percentage
bw = bwp * f0
t0 = 1 / f0
alpha = -(np.pi * bw / 2) ** 2 / np.log(np.sqrt(2) / 2)
s = -np.exp(-alpha * (t - t0) ** 2) * np.sin(2 * np.pi * f0 * (t - t0))

# plt.plot(t, s)

px = np.zeros((Nz, Nx), dtype=dtype)
px_1 = np.zeros((Nz, Nx), dtype=dtype)
pz = np.zeros((Nz, Nx), dtype=dtype)
pz_1 = np.zeros((Nz, Nx), dtype=dtype)
Ax = np.zeros((Nz, Nx), dtype=dtype)
Ax_1 = np.zeros((Nz, Nx), dtype=dtype)
Az = np.zeros((Nz, Nx), dtype=dtype)
Az_1 = np.zeros((Nz, Nx), dtype=dtype)

v = np.zeros((Nz, Nx))
v_1 = np. zeros((Nz, Nx))
v_2 = np. zeros((Nz, Nx))

# Laplacian Kernels Stencil Calculation - Prof. Dr. Pipa
deriv_order = 2
deriv_accuracy = 2
deriv_n_coef = 2 * np.floor((deriv_order + 1) / 2).astype('int') - 1 + deriv_accuracy
p = np.round((deriv_n_coef - 1) / 2).astype('int')
A = np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[None].T
b = np.zeros(2 * p + 1)
b[deriv_order] = math.factorial(deriv_order)
lapcoeff = np.zeros((deriv_n_coef, deriv_n_coef))
# Solve system A*w = b
lapcoeff[deriv_n_coef // 2, :] = np.linalg.solve(A, b)
lapcoeff += lapcoeff.T

# px[Nz // 2, Nx // 2] = 1
# pz[Nz // 2, Nx // 2] = 1
# Ax[Nz // 2, Nx // 2] = 1
# Az[Nz // 2, Nx // 2] = 1

d0 = 3 * c * np.log(1 / R) / (2 * Lp ** 3)
dx = np.zeros((Nz, Nx))
dz = np.zeros((Nz, Nx))
x = np.linspace(0, Lp, Np)
dx[:, -Np:] = d0 * x ** 2
dx[:, :Np] = d0 * (Lp - x) ** 2
dz[-Np:, :] = d0 * x[np.newaxis].T ** 2


# dz[:Np,:] = d0*(Lp-x[np.newaxis].T)**2

# d = np.array([[1]])
# d = np.array([[9/8, -1/24]])
# d = np.array([[75/64, -25/384, 3/640]])
# d = np.array([[75/64, -25/384, 3/640]])

# 10.1111/j.1365-246X.2009.04305.x
def coeff(N):
    c = np.zeros(N)
    for n in range(1, N + 1):
        m = np.arange(1, N + 1)
        m = m[m != n]
        p = np.prod(np.abs((2 * m - 1) ** 2 / ((2 * n - 1) ** 2 - (2 * m - 1) ** 2)))
        c[n - 1] = (((-1) ** (n + 1)) / (2 * n - 1)) * p

    return c


d = coeff(3)[None]
d = np.hstack((-np.flip(d), d))

print(d)

for k in range(1, Nt):
    px_1, pz_1, Ax_1, Az_1 = px, pz, Ax, Az

    v_1, v_2 = v.copy(), v_1.copy()

    # px[:,1:] = (1-dx[:,1:]*Dt)*px_1[:,1:] + c**2*(Dt/Dx)*(Ax_1[:,1:] - Ax_1[:, :-1])/2
    # pz[1:,:] = (1-dz[1:,:]*Dt)*pz_1[1:,:] + c**2*(Dt/Dx)*(Az_1[1:, :] - Az_1[:-1, :])/2
    # Ax[:,:-1] = (1-dx[:,:-1]*Dt)*Ax_1[:,:-1] + (Dt/Dx)*(px[:,1:]-px[:,:-1]+pz[:,1:]-pz[:,:-1])/2
    # Az[:-1,:] = (1-dz[:-1,:]*Dt)*Az_1[:-1,:] + (Dt/Dx)*(px[1:,:]-px[:-1,:]+pz[1:,:]-pz[:-1,:])/2

    # px[:,1:-1] = (1-dx[:,1:-1]*Dt)*px_1[:,1:-1] + c**2*(Dt/Dx)*(-Ax_1[:,2:] + Ax_1[:, :-2] - 8*Ax_1[:,1:-1] + 8*Ax_1[:,1:])/12
    # pz[1:-1,:] = (1-dz[1:-1,:]*Dt)*pz_1[1:-1,:] + c**2*(Dt/Dx)*(-Az_1[2:, :] + Az_1[:-2, :])/12
    # Ax[:,1:-1] = (1-dx[:,1:-1]*Dt)*Ax_1[:,1:-1] + (Dt/Dx)*(-px[:,2:]+px[:,:-2]-pz[:,2:]+pz[:,:-2])/12
    # Az[1:-1,:] = (1-dz[1:-1,:]*Dt)*Az_1[1:-1,:] + (Dt/Dx)*(-px[2:,:]+px[:-2,:]-pz[2:,:]+pz[:-2,:])/12

    px = (1 - dx * Dt) * px_1 + c ** 2 * (Dt / Dx) * correlate(Ax_1, d, mode='constant')
    pz = (1 - dz * Dt) * pz_1 + c ** 2 * (Dt / Dx) * correlate(Az_1, d.T, mode='constant')
    Ax = (1 - dx * Dt) * Ax_1 + (Dt / Dx) * correlate(px + pz, d, mode='constant', origin=[0, -1])
    Az = (1 - dz * Dt) * Az_1 + (Dt / Dx) * correlate(px + pz, d.T, mode='constant', origin=[-1, 0])

    px[Nz//2, Nx//2] += s[k]
    pz[Nz//2, Nx//2] += s[k]
    p = px + pz

    lap = correlate(v_1[:, :], lapcoeff)
    v = 2 * v_1 - v_2 + C2 * lap

    x = np.concatenate((p, v), 1)

    riw.setImage((px + pz).T, levels=[-.01, .01])
    app.processEvents()  ## force complete redraw for every plot
    plt.pause(1e-9)

app.quit()
