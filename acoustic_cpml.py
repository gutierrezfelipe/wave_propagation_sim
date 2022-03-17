#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com PML
# Utilizando CPML
# Felipe Derewlany Gutierrez


from PyQt5.QtWidgets import *
import sys
import numpy as np
from scipy import signal
import time
import math
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *


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
        self.setGeometry(800, 50, 800, 800)

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
Lx = 0.01  # [m]
Lz = 0.01  # [m]
dt = 5e-9  # [s/iteration]
dx = 10e-5  # [m/pixel]
dz = dx  # [m/pixel]
Nt = math.ceil(T / dt)


# Convolutional Perfectly Matched Layers size
CPML_size = 1
Nx = math.ceil(Lx / dx) + 2 * CPML_size
Nz = math.ceil(Lz / dz) + 2 * CPML_size

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

#soundspeed = 1000  # [m/s]
#soundspeed = 1481  # [m/s]
soundspeed = 2500  # [m/s]
#soundspeed = 3000  # [m/s]
#soundspeed = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

c = soundspeed / ad

CFL = soundspeed * dt / dx
print(f"CFL condition = {CFL}")

# Pressure Fields
u = np.zeros((Nz, Nx))
u_1 = np.zeros((Nz, Nx))
u_2 = np.zeros((Nz, Nx))

# Signal acquisitors
u_at_transducer = np.zeros(Nt)

# Sources
z_f = round(Nz * 0.2)  # Transductor z coordinate
x_f = round(Nx / 2)  # Transcuctor x coordinate
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 2e6  # [Hz]
delay = 1e-6
bandwidth = 0.6

f = signal.gausspulse(t - delay, frequency, bandwidth)

# Laplacian Kernels Stencil Calculation - Prof. Pipa
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


# Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()

# Start timer for simulation
start_time = time.time()

for k in range(4, Nt):
    iteration_start = time.time()

    u_0 = u
    u_2 = u_1
    u_1 = u_0

    """
    lap = signal.correlate(u_1[:, :], coeff, mode='same')
    u = 2 * u_1[:, :] - u_2[:, :] + (c ** 2) * lap
    """

    for j in range(CPML_size, Nz-CPML_size):
        for i in range(CPML_size, Nx-CPML_size):
            lap = u_1[j, i+1] + u_1[j+1, i] - 4 * u_1[j, i] + u_1[j-1, i] + u_1[j, i-1]
            u[j, i] = 2 * u_1[j, i] - u_2[j, i] + (c ** 2) * lap

    u[z_f, x_f] += f[k]

    # Signal Acquisition
    u_at_transducer[k] = u[z_f, x_f]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")

    # Exhibition Update - QT
    window.imv.setImage(u.T, levels=[-0.1, 0.1])
    App.processEvents()


App.exit()

end_time = time.time()
total_time = end_time - start_time

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

