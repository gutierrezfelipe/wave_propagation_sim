#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com PML

# Felipe Derewlany Gutierrez


from PyQt5.QtWidgets import *
import sys
import numpy as np
from scipy.ndimage import correlate
from scipy import signal
import laplaciano
import time
import math
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import copy


# Image View class
class ImageView(pg.ImageView):
    # constructor which inherit original
    # ImageView
    def __init__(self, *args, **kwargs):
        pg.ImageView.__init__(self, *args, **kwargs)


# RawImageWidget class
class RawImageWidget(pg.widgets.RawImageWidget.RawImageGLWidget):
    # constructor which inherit original
    # RawImageWidget
    def __init__(self):
        pg.widgets.RawImageWidget.RawImageGLWidget.__init__(self)


# Window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")

        # setting geometry
        self.setGeometry(200, 50, 800, 800)
        # 200, 50, 800, 800

        # setting animation
        self.isAnimated()

        # setting image
        self.image = np.random.normal(size=(500, 500))

        # showing all the widgets
        self.show()

        # creating a widget object
        self.widget = QWidget()

        # setting configuration options
        pg.setConfigOptions(antialias=True)

        # creating image view object
        self.imv = RawImageWidget()

        # setting image to image view
        self.imv.setImage(self.image, levels=[-0.1, 0.1])

        # Creating a grid layout
        self.layout = QGridLayout()

        # setting this layout to the widget
        self.widget.setLayout(self.layout)

        # plot window goes on right side, spanning 3 rows
        self.layout.addWidget(self.imv, 0, 0, 4, 1)

        # setting this widget as central widget of the main window
        self.setCentralWidget(self.widget)


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


def coeff2ndOrder(N):
    # Laplacian Kernels Stencil Calculation - Prof. Dr. Pipa
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


# Simulation Parameters
T = 0.00001  # [s]
Lx = 0.04  # [m]
Lz = 0.04  # [m]
dt = 5e-9  # [s/iteration]
dx = 10e-5  # [m/pixel]
dz = dx  # [m/pixel]
dh = dz  # [m/pixel]
Nt = math.ceil(T / dt)

# Perfectly Matched Layers size
PML_size = 20
Nx = math.ceil(Lx / dx) + 2 * PML_size
Nz = math.ceil(Lz / dz) + 2 * PML_size

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

# soundspeed = 1000  # [m/s]
# soundspeed = 1481  # [m/s]
# soundspeed = 2500  # [m/s]
# soundspeed = 3000  # [m/s]
soundspeed = 5800  # [m/s]
# soundspeed = 6000  # [m/s]

c = soundspeed * np.ones((Nz, Nx))
C2 = soundspeed ** 2 * dt ** 2 / dh ** 2

CFL = soundspeed * dt / dx
print(f"CFL condition = {CFL}")

# Pressure Fields
px = np.zeros((Nz, Nx))
px_1 = np.zeros((Nz, Nx))
pz = np.zeros((Nz, Nx))
pz_1 = np.zeros((Nz, Nx))
vx = np.zeros((Nz, Nx))
vx_1 = np.zeros((Nz, Nx))
vz = np.zeros((Nz, Nx))
vz_1 = np.zeros((Nz, Nx))
initial = np.zeros((Nz, Nx))
px_integral = np.zeros((Nz, Nx))
pz_integral = np.zeros((Nz, Nx))

px_integral = 1/2 * initial
pz_integral = 1/2 * initial


p = np.zeros((Nz, Nx))
p_1 = np.zeros((Nz, Nx))
p_2 = np.zeros((Nz, Nx))

# Signal acquisitors
u_at_transducer = np.zeros(Nt)

# Source config
z_f = round(Nz * 0.5)  # Transductor z coordinate
x_f = round(Nx * 0.5)  # Transcuctor x coordinate
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 2e6  # [Hz]
delay = 1e-6
bandwidth = 0.6
f = signal.gausspulse(t - delay, frequency, bandwidth)

source_integral = np.zeros(Nt)


# PML parameters
Li = PML_size * dh
R = 1e-7
Vmax = soundspeed
# d0 = -3 / (2 * Li) * math.log(R, 10)

ones = np.ones((PML_size, Nx))

fade = np.linspace(0, Li, PML_size)
f_i_bottom = np.outer(fade, ones)
f_i_right = f_i_bottom.T

fade = np.linspace(Li, 0, PML_size)
f_i_top = np.outer(fade, ones)
f_i_left = f_i_top.T

f_i = np.zeros((Nz, Nx))
f_i[0:PML_size, :] += f_i_top[:, :Nx]
f_i[Nz - PML_size:Nz, :] += f_i_bottom[:, :Nx]
f_i[:, 0:PML_size] += f_i_left[:Nz, :]
f_i[:, Nx - PML_size:Nx] += f_i_right[:Nz, :]

d_x = 3 * Vmax / (2 * Li) * (f_i / Li) ** 2 * math.log(1 / R, 10)  # omega_x
d_z = 3 * Vmax / (2 * Li) * (f_i / Li) ** 2 * math.log(1 / R, 10)  # omega_z

# constants
a_x = 1
a_z = 1

rho = 1024 * np.ones((Nz, Nx))  # [Kg/m³]
gamma = 2e-3  # [s/m²]

D_xv = a_x/dt + (a_x * gamma * c ** 2 + d_x)/2 + d_x * gamma * c ** 2 * dt / 2
D_zv = a_z/dt + (a_z * gamma * c ** 2 + d_z)/2 + d_z * gamma * c ** 2 * dt / 2

f_1x = (a_x/dt - d_x/2) / (a_x/dt + d_x/2)
f_1z = (a_z/dt - d_z/2) / (a_z/dt + d_z/2)
f_2x = 1 / ((a_x/dt + d_x/2) * rho * dx)
f_2z = 1 / ((a_z/dt + d_z/2) * rho * dz)
f_3x = (a_x/dt - (a_x * gamma * c ** 2 + d_x)/2) / D_xv
f_3z = (a_z/dt - (a_z * gamma * c ** 2 + d_z)/2) / D_zv
f_4x = d_x * gamma * c ** 2 * dt / D_xv
f_4z = d_z * gamma * c ** 2 * dt / D_zv
f_5x = - rho * c ** 2 / D_xv
f_5z = - rho * c ** 2 / D_zv
f_6x = 1 / D_xv
f_6z = 1 / D_zv

# Deriv accuracy and coefficients
accuracy = 4
c1stOrd = coeff1storder(accuracy)
c2ndOrd = coeff2ndOrder(2*accuracy)

# Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()

# Start timer for simulation
start_time = time.time()

for k in range(1, Nt):
    iteration_start = time.time()

    px_1, pz_1, vx_1, vz_1 = px, pz, vx, vz

    # gao 2015
    # px = (1 - d_x * dt) * px_1 - c[:, :] ** 2 * (dt / dx) * correlate(vx_1, c1stOrd, mode='constant')
    # pz = (1 - d_z * dt) * pz_1 - c[:, :] ** 2 * (dt / dx) * correlate(vz_1, c1stOrd.T, mode='constant')
    # px[Nz // 2, Nx // 2] += f[k] / 2
    # pz[Nz // 2, Nx // 2] += f[k] / 2
    # vx = (1 - d_x * dt) * vx_1 - (dt / dx) * correlate(px + pz, c1stOrd, mode='constant', origin=[0, -1])
    # vz = (1 - d_z * dt) * vz_1 - (dt / dx) * correlate(px + pz, c1stOrd.T, mode='constant', origin=[-1, 0])

    # liu 1997
    px_integral += px
    pz_integral += pz

    px = f_3x * px_1 + f_4x * px_integral + f_5x * correlate(vx_1, c1stOrd, mode='constant')
    pz = f_3z * pz_1 + f_4z * pz_integral + f_5z * correlate(vz_1, c1stOrd.T, mode='constant')

    px[z_f, x_f] += f_6x[z_f, x_f] * f[k] / 2
    pz[z_f, x_f] += f_6z[z_f, x_f] * f[k] / 2

    vx = f_1x * vx_1 + f_2x * correlate(px+pz, c1stOrd, mode='constant', origin=[0, -1])
    vz = f_1z * vz_1 + f_2z * correlate(px+pz, c1stOrd.T, mode='constant', origin=[-1, 0])

    # without pml
    p_1, p_2 = p, p_1
    p = 2 * p_1 - p_2 + (dt * c[:, :] / dx) ** 2 * correlate(p_1, c2ndOrd, mode='constant')
    p[Nz // 2, Nx // 2] += f[k]

    # Signal Acquisition
    u_at_transducer[k] = px[z_f, x_f] + pz[z_f, x_f]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt-1} - Math Time: {math_time - iteration_start} s")
    # print(np.shares_memory(u, uu))

    # Exhibition Update - QT
    # x = np.concatenate(((px+pz), p), 1)
    # window.imv.setImage(x.T, levels=[-0.1, 0.1])
    window.imv.setImage((px + pz).T, levels=[-0.1, 0.1])
    App.processEvents()

App.exit()

end_time = time.time()
total_time = end_time - start_time

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")
