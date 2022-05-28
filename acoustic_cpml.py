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
T = 0.000008  # [s]
Lx = 0.02  # [m]
Lz = 0.02  # [m]
dt = 5e-9  # [s/iteration]
dx = 10e-5  # [m/pixel]
dz = dx  # [m/pixel]
dh = dz  # [m/pixel]
Nt = math.ceil(T / dt)


# Convolutional Perfectly Matched Layers size
CPML_size = 20
Nx = math.ceil(Lx / dx) + 2 * CPML_size
Nz = math.ceil(Lz / dz) + 2 * CPML_size

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

#soundspeed = 1000  # [m/s]
#soundspeed = 1481  # [m/s]
soundspeed = 2500  # [m/s]
#soundspeed = 3000  # [m/s]
#soundspeed = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

#c = soundspeed / ad
c = soundspeed
C2 = c**2 * dt**2 / dh**2

CFL = soundspeed * dt / dx
print(f"CFL condition = {CFL}")

# Pressure Fields
u = np.zeros((Nz, Nx))
u_1 = np.zeros((Nz, Nx))
u_2 = np.zeros((Nz, Nx))

v = np.zeros((Nz, Nx))
v_1 = np. zeros((Nz, Nx))
v_2 = np. zeros((Nz, Nx))

#mask = np.full((Nz, Nx), True)
#mask[0:CPML_size, :] = False
#mask[Nz-CPML_size:Nz, :] = False
#mask[:, 0:CPML_size] = False
#mask[:, Nx-CPML_size:Nx] = False

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

# CPML parameters
Li = CPML_size * dh
R = 0.0000001
Vmax = 2500
#d0 = -3 / (2 * Li) * math.log(R, 10)

ones = np.ones((CPML_size, Nx))

fade = np.linspace(0, Li, CPML_size)
f_i_bottom = np.outer(fade, ones)
f_i_right = f_i_bottom.T

fade = np.linspace(Li, 0, CPML_size)
f_i_top = np.outer(fade, ones)
f_i_left = f_i_top.T

f_i = np.zeros((Nz, Nx))
f_i[0:CPML_size, :] += f_i_top[:, :Nx]
f_i[Nz-CPML_size:Nz, :] += f_i_bottom[:, :Nx]
f_i[:, 0:CPML_size] += f_i_left[:Nz, :]
f_i[:, Nx-CPML_size:Nx] += f_i_right[:Nz, :]

#plt.imshow(f_i)
#plt.show()

sigmai = np.zeros((Nz, Nx))
alphai = np.zeros((Nz, Nx))
ai = np.zeros((Nz, Nx))
bi = np.zeros((Nz, Nx))

sigmai = 3 * Vmax / (2 * Li) * (f_i/Li) ** 2 * math.log(1/R, 10)  # dx(x)
alphai = math.pi * frequency * ((Li - f_i)/Li)

ai = math.e ** -((sigmai + alphai) * dt)
bi = (sigmai / (sigmai - alphai)) * (ai - 1)

psi = np.zeros((Nz, Nx))
psii0 = np.zeros((Nz, Nx))
psix = np.zeros((Nz, Nx))
psiz = np.zeros((Nz, Nx))
psix_1 = np.zeros((Nz, Nx))
psiz_1 = np.zeros((Nz, Nx))

psix[:, 1:-1] = ai[:, 1:-1] * psii0[:, 1:-1] + (u_1[:, 2:] - u_1[:, 0:-2])/2
psiz[1:-1, :] = ai[1:-1, :] * psii0[1:-1, :] + (u_1[2:, :] - u_1[0:-2, :])/2

zeta = np.zeros((Nz, Nx))
zetai0 = np.zeros((Nz, Nx))
zetax = np.zeros((Nz, Nx))
zetaz = np.zeros((Nz, Nx))
zetax_1 = np.zeros((Nz, Nx))
zetaz_1 = np.zeros((Nz, Nx))

zetax[:, 2:] = ai[:, 1:-1] * zetai0[:, 1:-1] + bi[:, 1:-1] * \
                 ((u_1[:, 0:-2] - 2 * u_1[:, 1:-1] + u_1[:, 2:]) + (psix[:, 2:] - psix[:, 0:-2])/2)
zetaz[:-2, :] = ai[1:-1, :] * zetai0[1:-1, :] + bi[1:-1, :] * \
                 ((u_1[0:-2, :] - 2 * u_1[1:-1, :] + u_1[2:, :]) + (psiz[2:, :] - psiz[0:-2, :])/2)

# Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()

# Start timer for simulation
start_time = time.time()

for k in range(3, Nt):
    iteration_start = time.time()

    #u_2 = u_1
    u_1, u_2 = u.copy(), u_1.copy()
    v_1, v_2 = v.copy(), v_1.copy()
    #uu_1, uu_2 = uu, uu_1

    psix_1 = psix
    psiz_1 = psiz
    psix[:, 2:] = ai[:, :-2] * psix_1[:, :-2] + (u_1[:, 2:] - u_1[:, 0:-2]) / 2
    psiz[:-2, :] = ai[2:, :] * psiz_1[2:, :] + (u_1[2:, :] - u_1[0:-2, :]) / 2

    zetax_1 = zetax
    zetaz_1 = zetaz
    zetax[:, :-2] = ai[:, 1:-1] * zetax_1[:, 1:-1] + bi[:, 1:-1] * \
                     ((u_1[:, 0:-2] - 2 * u_1[:, 1:-1] + u_1[:, 2:]) + (psix[:, 2:] - psix[:, 0:-2]) / 2)
    zetaz[:-2, :] = ai[1:-1, :] * zetaz_1[1:-1, :] + bi[1:-1, :] * \
                     ((u_1[0:-2, :] - 2 * u_1[1:-1, :] + u_1[2:, :]) + (psiz[2:, :] - psiz[0:-2, :]) / 2)

    #lap = signal.correlate(u_1[CPML_size:Nz-CPML_size, CPML_size:Nx-CPML_size], coeff, mode='same')
    #u[CPML_size:Nz-CPML_size, CPML_size:Nx-CPML_size] = 2 * u_1[CPML_size:Nz-CPML_size, CPML_size:Nx-CPML_size] - u_2[CPML_size:Nz-CPML_size, CPML_size:Nx-CPML_size] + (c ** 2) * lap

    #lap = signal.correlate(u_1[mask], coeff, mode='same')
    lap = laplaciano.fdm_laplaciano(u_1, 2)
    #u = np.zeros_like(u)
    #u[mask] = (2 * u_1[mask] - u_2[mask] + (c ** 2) * lap[mask]).reshape((Nz, Nx))
    #temp = (2 * u_1[mask] - u_2[mask] + (c ** 2) * lap[mask]).reshape((Nz, Nx))

    psi[:, :-1] = psix[:, 1:] - psix[:, :-1]
    psi[1:, :] += psiz[1:, :] - psiz[:-1, :]

    zeta = zetax + zetaz

    u[:, :] = 2 * u_1[:, :] - u_2[:, :] + C2 * lap + \
              C2 * dh * psi + C2 * dh**2 * zeta

    lap = laplaciano.fdm_laplaciano(v_1, 2)
    v = 2 * v_1 - v_2 + C2 * lap

    #plt.imshow(u-temp)
    #plt.show()
    #print(u.shape)
    #np.copyto(u, 2 * u_1 - u_2 + (c ** 2) * lap)
    #uu = 2 * uu_1 - uu_2 + (c ** 2) * lap

    u[z_f, x_f] = f[k] + u[z_f, x_f]
    v[z_f, x_f] = f[k] + v[z_f, x_f]

    # Signal Acquisition
    u_at_transducer[k] = u[z_f, x_f]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")
    #print(np.shares_memory(u, uu))

    # Exhibition Update - QT
    x = np.concatenate((u, v), 1)
    #window.imv.setImage(u.T, levels=[-0.1, 0.1])
    window.imv.setImage(x.T, levels=[-0.1, 0.1])
    App.processEvents()


App.exit()

end_time = time.time()
total_time = end_time - start_time

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")


