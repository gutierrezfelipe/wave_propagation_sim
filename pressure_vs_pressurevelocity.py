#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:08:26 2022

@author: danielpipa
"""

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from scipy.ndimage import correlate
import math

dtype = 'float32'  # float32 is the fastest

Dx = .1e-3
Dz = Dx
Dt = 5e-9

# PML
# R, Np = 1e-5, 10
# R, Np = 1e-5, 20
# # R, Np = 1e-2, 30
# Lp = Np * Dx

Lx = 50e-3  # Block width
Lz = 50e-3  # Block height
Lt = 10e-6  # Simulation time

Nx = round(Lx / Dx)
Nz = round(Lz / Dz)
Nt = round(Lt / Dt)

# c = 5490
# c = 1480
c = 5490 * np.ones((Nz, Nx))

radius = 0.030/2/Dx  # 30mm diameter circle
z_center = Nz//2
x_center = Nx//2

for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2:
            c[j, i] = 2760


z_f = 5
x_f = Nx // 2
p_recording = np.zeros(Nt)
pv_recording = np.zeros(Nt)

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

# 10.1111/j.1365-246X.2009.04305.x
def coeff1storder(N):
    c = np.zeros(N)
    for n in range(1, N + 1):
        m = np.arange(1, N + 1)
        m = m[m != n]
        p = np.prod(np.abs((2 * m - 1) ** 2 / ((2 * n - 1) ** 2 - (2 * m - 1) ** 2)))
        c[n - 1] = (((-1) ** (n + 1)) / (2 * n - 1)) * p

    c = c[None]
    return np.hstack((-np.flip(c), c))


accuracy = 4

c1stOrd = coeff1storder(accuracy)


def coeff2ndOrder(N):
    deriv_order = 2
    deriv_n_coef = 2 * np.floor((deriv_order + 1) / 2).astype('int') - 1 + N
    p = np.round((deriv_n_coef - 1) / 2).astype('int')
    A = np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[None].T
    b = np.zeros(2 * p + 1)
    b[deriv_order] = math.factorial(deriv_order)
    h = np.zeros((deriv_n_coef, deriv_n_coef))
    h[deriv_n_coef // 2, :] = np.linalg.solve(A, b)
    h += h.T
    return h


c2ndOrd = coeff2ndOrder(2*accuracy)


px = np.zeros((Nz, Nx), dtype=dtype)
px_1 = np.zeros((Nz, Nx), dtype=dtype)
pz = np.zeros((Nz, Nx), dtype=dtype)
pz_1 = np.zeros((Nz, Nx), dtype=dtype)
Ax = np.zeros((Nz, Nx), dtype=dtype)
Ax_1 = np.zeros((Nz, Nx), dtype=dtype)
Az = np.zeros((Nz, Nx), dtype=dtype)
Az_1 = np.zeros((Nz, Nx), dtype=dtype)

p = np.zeros((Nz, Nx), dtype=dtype)
p_1 = np.zeros((Nz, Nx), dtype=dtype)
p_2 = np.zeros((Nz, Nx), dtype=dtype)

for k in range(1, Nt):
    px_1, pz_1, Ax_1, Az_1 = px, pz, Ax, Az

    px = px_1 - c ** 2 * (Dt / Dx) * correlate(Ax_1, c1stOrd, mode='constant')
    pz = pz_1 - c ** 2 * (Dt / Dx) * correlate(Az_1, c1stOrd.T, mode='constant')
    px[z_f, x_f] += s[k]/2
    pz[z_f, x_f] += s[k]/2
    Ax = Ax_1 - (Dt / Dx) * correlate(px + pz, c1stOrd, mode='constant', origin=[0, -1])
    Az = Az_1 - (Dt / Dx) * correlate(px + pz, c1stOrd.T, mode='constant', origin=[-1, 0])
    pv_recording[k] = px[Nz // 2, Nx // 2] + pz[Nz // 2, Nx // 2]

    p_1, p_2 = p, p_1
    p = 2*p_1 - p_2 + (Dt*c/Dx)**2 * correlate(p_1, c2ndOrd, mode='constant')
    p[z_f, x_f] += s[k]
    p_recording[k] = p[Nz // 2, Nx // 2]

    riw.setImage(np.vstack((p.T, (px + pz).T)), levels=[-.1, .1])
    app.processEvents()  ## force complete redraw for every plot
    plt.pause(1e-12)

app.quit()

plt.figure()
plt.plot(p[Nz//2, :], label='pressure')
plt.plot(px[Nz//2, :]+pz[Nz//2, :], label='pressure velocity')
plt.legend()

plt.figure()
plt.plot(t, p_recording, label='pressure')
plt.plot(t, pv_recording, label='pressure velocity')
plt.legend()


plt.show()

