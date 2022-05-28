#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com PML
# Utilizando CPML
# Felipe Derewlany Gutierrez


from PyQt5.QtWidgets import *
import sys
import numpy as np
#import cupy as np
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
        self.setGeometry(200, 50, 1600, 800)
        #200, 50, 800, 800

        # setting animation
        self.isAnimated()

        #setting image
        self.image = np.random.normal(size=(500, 500))

        # showing all the widgets
        self.show()

        # creating a widget object
        self.widget = QWidget()

        # setting configuration options
        pg.setConfigOptions(antialias=True)

        # creating image view view object
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


# Simulation Parameters
T = 0.00001  # [s]
Lx = 0.02  # [m]
Lz = 0.02  # [m]
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

#soundspeed = 1000  # [m/s]
#soundspeed = 1481  # [m/s]
soundspeed = 2500  # [m/s]
#soundspeed = 3000  # [m/s]
#soundspeed = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

c = soundspeed
C2 = c**2 * dt**2 / dh**2

CFL = soundspeed * dt / dx
print(f"CFL condition = {CFL}")

# Pressure Fields
px = np.zeros((Nz, Nx))
px_1 = np.zeros((Nz, Nx))
px_2 = np.zeros((Nz, Nx))
pz = np.zeros((Nz, Nx))
pz_1 = np.zeros((Nz, Nx))
pz_2 = np.zeros((Nz, Nx))
u_accuracy_1 = np.zeros((Nz, Nx))
u_accuracy_2 = np.zeros((Nz, Nx))
aux = np.zeros((Nz, Nx))

Ax = np.zeros((Nz, Nx))
Ax_1 = np.zeros((Nz, Nx))
Ax_2 = np.zeros((Nz, Nx))
Az = np.zeros((Nz, Nx))
Az_1 = np.zeros((Nz, Nx))
Az_2 = np.zeros((Nz, Nx))

v = np.zeros((Nz, Nx))
v_1 = np. zeros((Nz, Nx))
v_2 = np. zeros((Nz, Nx))

# Signal acquisitors
u_at_transducer = np.zeros(Nt)

# Source config
z_f = round(Nz * 0.5)  # Transductor z coordinate
x_f = round(Nx / 2)  # Transcuctor x coordinate
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 2e6  # [Hz]
delay = 1e-6
bandwidth = 0.6
f = signal.gausspulse(t - delay, frequency, bandwidth)

# Laplacian Kernels Stencil Calculation - Prof. Dr. Pipa
deriv_order = 2
deriv_accuracy = 2
deriv_n_coef = 2 * np.floor((deriv_order + 1) / 2).astype('int') - 1 + deriv_accuracy
p = np.round((deriv_n_coef - 1) / 2).astype('int')
A = np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[None].T
b = np.zeros(2 * p + 1)
b[deriv_order] = math.factorial(deriv_order)
coeff = np.zeros((deriv_n_coef, deriv_n_coef))
# Solve system A*w = b
coeff[deriv_n_coef // 2, :] = np.linalg.solve(A, b)
coeff += coeff.T

# PML parameters
Li = PML_size * dh
R = 0.0000001
Vmax = 2500
#d0 = -3 / (2 * Li) * math.log(R, 10)

ones = np.ones((PML_size, Nx))

fade = np.linspace(0, Li, PML_size)
f_i_bottom = np.outer(fade, ones)
f_i_right = f_i_bottom.T

fade = np.linspace(Li, 0, PML_size)
f_i_top = np.outer(fade, ones)
f_i_left = f_i_top.T

f_i = np.zeros((Nz, Nx))
f_i[0:PML_size, :] += f_i_top[:, :Nx]
f_i[Nz-PML_size:Nz, :] += f_i_bottom[:, :Nx]
f_i[:, 0:PML_size] += f_i_left[:Nz, :]
f_i[:, Nx-PML_size:Nx] += f_i_right[:Nz, :]

d_x = np.zeros((Nz, Nx))
d_z = np.zeros((Nz, Nx))
d_x = 3 * Vmax / (2 * Li) * (f_i/Li) ** 2 * math.log(1/R, 10)  # dx(x)
d_z = 3 * Vmax / (2 * Li) * (f_i/Li) ** 2 * math.log(1/R, 10)  # dz(z)

# Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()

# Start timer for simulation
start_time = time.time()

for k in range(3, Nt):
    iteration_start = time.time()

    px_1, px_2 = px.copy(), px_1.copy()
    pz_1, pz_2 = pz.copy(), pz_1.copy()
    #pz_1 = pz

    Ax_1, Ax_2 = Ax.copy(), Ax_1.copy()
    Az_1, Az_2 = Az.copy(), Az_1.copy()
    #Az_1 = Az

    v_1, v_2 = v.copy(), v_1.copy()

    ## PV acoustic wave equation formulation

    '''
    # Derivada Temporal: Forward Acurácia 1
    # Derivada Espacial: Forward Acurácia 1
    px[:, 1:] = px_1[:, 1:] - d_x[:, 1:] * px_1[:, 1:] * dt + c**2 *dt/(dx) * (Ax[:, 1:] - Ax[:, :-1])
    pz[:-1, :] = pz_1[:-1, :] - d_z[:-1, :] * pz_1[:-1, :] * dt + c**2 *dt/(dz) * (Az[1:, :] - Az[:-1, :])

    Ax[:, :-1] = Ax_1[:, :-1] - d_x[:, :-1] * Ax_1[:, :-1]*dt + \
                  dt/dx * (px[:, 1:] - px[:, :-1] + pz[:, 1:] - pz[:, :-1])

    Az[1:, :] = Az_1[1:, :] - d_z[1:, :] * Az_1[1:, :]*dt + \
                  dt/dz * (px[1:, :] - px[:-1, :] + pz[1:, :] - pz[:-1, :])

    px[z_f, x_f] = f[k] + px[z_f, x_f]
    pz[z_f, x_f] = f[k] + pz[z_f, x_f]

    u_accuracy_1 = px + pz
    
    # Derivada Temporal: Forward Acurácia 1
    # Derivada Espacial: Central Acurácia 1
    px[:, 2:] = px_1[:, 2:] - d_x[:, 2:] * px_1[:, 2:] * dt + c ** 2 * dt / (2 * dx) * (Ax[:, 2:] - Ax[:, :-2])
    pz[:-2, :] = pz_1[:-2, :] - d_z[:-2, :] * pz_1[:-2, :] * dt + c ** 2 * dt / (2 * dz) * (Az[2:, :] - Az[:-2, :])

    Ax[:, :-2] = Ax_1[:, :-2] - d_x[:, :-2] * Ax_1[:, :-2] * dt + \
                 dt / (2 *dx) * (px[:, 2:] - px[:, :-2] + pz[:, 2:] - pz[:, :-2])

    Az[2:, :] = Az_1[2:, :] - d_z[2:, :] * Az_1[2:, :] * dt + \
                dt / (2 * dz) * (px[2:, :] - px[:-2, :] + pz[2:, :] - pz[:-2, :])

    px[z_f, x_f] = f[k] + px[z_f, x_f]
    pz[z_f, x_f] = f[k] + pz[z_f, x_f]

    u_accuracy_1_c = px + pz
    '''

    # Derivada Temporal: forward accuracy 1
    # Derivada Espacial: central accuracy 2
    aux[:, 2:] = 9/8 * Ax[:, :-2] - 9/8 * Ax[:, 2:]
    aux[:, 1:] += 1/24 * Ax[:, 1:] - 1/24 * Ax[:, :-1]

    px[:, 2:] = px_1[:, 2:] - d_x[:, 2:] * px_1[:, 2:] * dt + c ** 2 * dt / (12 * dx) * aux[:, 2:]

    aux[:-2, :] = 9/8 * Az[:-2, :] - 9/8 * Az[2:, :]
    aux[:-1, :] += 1/24 * Az[1:, :] - 1/24 * Az[:-1, :]

    pz[:-2, :] = pz_1[:-2, :] - d_z[:-2, :] * pz_1[:-2, :] * dt + c ** 2 * dt / (12 * dz) * aux[:-2, :]

    aux[:, :-2] = px[:, :-2] - px[:, 2:] + pz[:, :-2] - pz[:, 2:]
    aux[:, :-1] += 8 * px[:, 1:] - 8 * px[:, :-1] + 8 * pz[:, 1:] - 8 * pz[:, :-1]

    Ax[:, :-2] = Ax_1[:, :-2] - d_x[:, :-2] * Ax_1[:, :-2] * dt + \
                 dt / (12 * dx) * aux[:, :-2]

    aux[2:, :] = px[:-2, :] - px[2:, :] + pz[:-2, :] - pz[2:, :]
    aux[1:, :] += 8 * px[1:, :] - 8 * px[:-1, :] + 8 * pz[1:, :] - 8 * pz[:-1, :]

    Az[2:, :] = Az_1[2:, :] - d_z[2:, :] * Az_1[2:, :] * dt + \
                dt / (12 * dz) * aux[2:, :]

    px[z_f, x_f] = f[k] + px[z_f, x_f]
    pz[z_f, x_f] = f[k] + pz[z_f, x_f]

    u_accuracy_12 = px + pz

    '''
    # Derivada Temporal: Forward Acurácia 2
    # Derivada ESpacial: Central Acurácia 2
    aux[:, 2:] = Ax[:, :-2] - Ax[:, 2:]
    aux[:, 1:] += 8 * Ax[:, 1:] - 8 * Ax[:, :-1]

    px[:, 2:] = 4 * px_1[:, 2:] + (2 * d_x[:, 2:] * dt - 3) * px_2[:, 2:] - \
                2 * dt * c ** 2 / (12 * dx) * aux[:, 2:]

    aux[:-2, :] = Az[:-2, :] - Az[2:, :]
    aux[:-1, :] += 8 * Az[1:, :] - 8 * Az[:-1, :]

    pz[:-2, :] = 4 * pz_1[:-2, :] + (2 * d_z[:-2, :] * dt - 3) * pz_2[:-2, :] - \
                 2 * dt * c ** 2 / (12 * dz) * aux[:-2, :]

    aux[:, :-2] = px[:, :-2] - px[:, 2:] + pz[:, :-2] - pz[:, 2:]
    aux[:, :-1] += 8 * px[:, 1:] - 8 * px[:, :-1] + 8 * pz[:, 1:] - 8 * pz[:, :-1]

    Ax[:, :-2] = 4 * Ax_1[:, :-2] + (2 * d_x[:, :-2] * dt - 3) * Ax_2[:, :-2] - \
                2 * dt / (12 * dx) * aux[:, :-2]

    aux[2:, :] = px[:-2, :] - px[2:, :] + pz[:-2, :] - pz[2:, :]
    aux[1:, :] += 8 * px[1:, :] - 8 * px[:-1, :] + 8 * pz[1:, :] - 8 * pz[:-1, :]

    Az[2:, :] = 4 * Az_1[2:, :] + (2 * d_z[2:, :] * dt - 3) * Az_2[2:, :] - \
                2 * dt / (12 * dz) * aux[2:, :]

    px[z_f, x_f] = f[k] + px[z_f, x_f]
    pz[z_f, x_f] = f[k] + pz[z_f, x_f]

    u_accuracy_2 = px + pz
    '''

    # without pml
    lap = laplaciano.fdm_laplaciano(v_1, 2)
    v = 2 * v_1 - v_2 + C2 * lap

    v[z_f, x_f] = f[k] + v[z_f, x_f]

    # Signal Acquisition
    u_at_transducer[k] = u_accuracy_1[z_f, x_f]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")
    #print(np.shares_memory(u, uu))

    # Exhibition Update - QT
    x = np.concatenate((u_accuracy_12, v), 1)
    #window.imv.setImage(u_accuracy_1.T, levels=[-0.1, 0.1])
    window.imv.setImage(x.T, levels=[-0.1, 0.1])
    App.processEvents()


App.exit()

end_time = time.time()
total_time = end_time - start_time

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")
