#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com PML

# Felipe Derewlany Gutierrez


from PyQt5.QtWidgets import *
import sys
import numpy as np
from scipy.ndimage import correlate
from scipy import signal
import laplaciano
import matplotlib
import time
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

# Uses
matplotlib.use('TkAgg')

# Simulation Parameters
T = 12e-3  # [s]
Lx = 2.2  # [m]
Lz = 2.2  # [m]
dt = 3e-6  # [s/iteration]
dx = 2.5e-2  # [m/pixel]
dz = dx  # [m/pixel]
dh = dz  # [m/pixel]
Nt = round(T / dt)

# Perfectly Matched Layers size
PML_size = 12
Nx = math.ceil(Lx / dx) + 2 * PML_size
Nz = math.ceil(Lz / dz) + 2 * PML_size

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

# media setup
# soundspeed = 1000  # [m/s]
# soundspeed = 1481  # [m/s]
soundspeed = 1480  # [m/s]
# soundspeed = 3000  # [m/s]
# soundspeed = 5800  # [m/s]
# soundspeed = 6000  # [m/s]

c = soundspeed * np.ones((Nz, Nx))
# outer bounds zero (reflection correction)
c[0, :] = 0
c[Nz-1, :] = 0
c[:, 0] = 0
c[:, Nx-1] = 0

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

# segundo eq. (19) - inicializa
px_integral = 1/2 * initial
pz_integral = 1/2 * initial


p = np.zeros((Nz, Nx))
p_1 = np.zeros((Nz, Nx))
p_2 = np.zeros((Nz, Nx))

# Signal acquisiton
n_sensors = 32
sensors = np.zeros((n_sensors, Nt))
sensors_pos = np.zeros((2, n_sensors), dtype='int')
xzm = np.full((Nz, Nx), False)
z = round(Nz * 0.05)  # 56
xi = round(Nx * 0.25)  # 52
sensors_pos[0, :n_sensors//2] = z
# sensors_space = round(0.05/dx)

# reflection (top)
for i in range(n_sensors//2):
    # x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / n_sensors))
    # x = xi + (i-1) * sensors_space
    x = int(Nx // 3 + i * ((Nx - 1 * Nx // 6) / n_sensors))
    sensors_pos[1, i] = x
    xzm[z, x] = True

# middle (source) sensors
#x_s = round(Nx * 0.5)
z_s = round(Nz * 0.5)
sensors_pos[0, n_sensors//2:] = z_s
sensors_pos[1, n_sensors//2:] = sensors_pos[1, :n_sensors//2]
xzm[sensors_pos[0, :], sensors_pos[1, :]] = True

# Source config
z_f = round(Nz * 0.5)  # Transductor z coordinate
x_f = round(Nx * 0.5)  # Transcuctor x coordinate
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 2e3  # [Hz]
delay = 1.25e-3
bandwidth = 0.6
f = signal.gausspulse(t - delay, frequency, bandwidth)

#f -= np.mean(f)
source_integral = 0


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

# fading omega
omega_x = 0.0001 * np.ones((Nz, Nx))
omega_z = 0.0001 * np.ones((Nz, Nx))
pixels = np.arange(0, PML_size)
omega_max = soundspeed  # [?]
p = 2
fade = (PML_size - 1/2 - pixels) ** p / ((PML_size - 1/2) ** p) * omega_max

omega_x[:, 0:PML_size] += fade
omega_x[:, Nx - PML_size:Nx] += np.flip(fade)
omega_z[0:PML_size, :] += np.outer(fade, ones)[:, :Nx]
omega_z[Nz - PML_size:Nz, :] += np.flip(np.outer(fade, ones)[:, :Nx])

# constants
a_x = 1
a_z = 1

rho = 1200 * np.ones((Nz, Nx))  # [Kg/m³]
# rho = np.ones((Nz, Nx))  # [Kg/m³]
gamma = 2e-3  # [s/m²]
# gamma = 0  # [s/m²]
# gamma = 2e-3*np.ones((Nz, Nx))
#gamma[Nz//2, Nx//2] = 0


D_xv = a_x/dt + (a_x * gamma * c ** 2 + omega_x)/2 + omega_x * gamma * c ** 2 * dt / 2
D_zv = a_z/dt + (a_z * gamma * c ** 2 + omega_z)/2 + omega_z * gamma * c ** 2 * dt / 2

f_1x = (a_x/dt - omega_x/2) / (a_x/dt + omega_x/2)
f_1z = (a_z/dt - omega_z/2) / (a_z/dt + omega_z/2)
f_2x = - 1 / ((a_x/dt + omega_x/2) * rho * dx)
f_2z = - 1 / ((a_z/dt + omega_z/2) * rho * dz)
f_3x = (a_x/dt - (a_x * gamma * c ** 2 + omega_x)/2) / D_xv
f_3z = (a_z/dt - (a_z * gamma * c ** 2 + omega_z)/2) / D_zv
f_4x = - omega_x * gamma * c ** 2 * dt / D_xv
f_4z = - omega_z * gamma * c ** 2 * dt / D_zv
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
    px = (1 - d_x * dt) * px_1 - c[:, :] ** 2 * (dt / dx) * correlate(vx_1, c1stOrd, mode='constant')
    pz = (1 - d_z * dt) * pz_1 - c[:, :] ** 2 * (dt / dx) * correlate(vz_1, c1stOrd.T, mode='constant')
    px[Nz // 2, Nx // 2] += f[k] / 2
    pz[Nz // 2, Nx // 2] += f[k] / 2
    vx = (1 - d_x * dt) * vx_1 - (dt / dx) * correlate(px + pz, c1stOrd, mode='constant', origin=[0, -1])
    vz = (1 - d_z * dt) * vz_1 - (dt / dx) * correlate(px + pz, c1stOrd.T, mode='constant', origin=[-1, 0])

    # liu 1997
    # integrals
    # px_integral += px
    # pz_integral += pz
    #
    # source_integral += f[k]  # correction of unity
    #
    # px = f_3x * px_1 + f_4x * px_integral + f_5x * correlate(vx_1, c1stOrd, mode='constant')
    # pz = f_3z * pz_1 + f_4z * pz_integral + f_5z * correlate(vz_1, c1stOrd.T, mode='constant')
    #
    # px[z_f, x_f] = f_6x[z_f, x_f] * source_integral / 2
    # pz[z_f, x_f] = f_6z[z_f, x_f] * source_integral / 2
    #
    # # px[z_f, x_f] = source_integral / 2
    # # pz[z_f, x_f] = source_integral / 2
    #
    #
    # vx = f_1x * vx_1 + f_2x * correlate(px+pz, c1stOrd, mode='constant', origin=[0, -1])
    # vz = f_1z * vz_1 + f_2z * correlate(px+pz, c1stOrd.T, mode='constant', origin=[-1, 0])

    # # liu1997 my FDM discretization - rho inside Ax, Az instead of vx, vz
    # vx = (1 - omega_x * dt) * vx_1 - dt/dx * correlate(px + pz, c1stOrd, mode='constant', origin=[0, -1])
    # vz = (1 - omega_z * dt) * vz_1 - dt/dz * correlate(px + pz, c1stOrd.T, mode='constant', origin=[-1, 0])
    #
    # px = (1 - (gamma * c**2 + omega_x) * dt) * px_1 + (omega_x * gamma * c ** 2 * dt) * px_integral - \
    #      (c ** 2 * dt/dx) * correlate(vx_1, c1stOrd, mode='constant')
    # pz = (1 - (gamma * c**2 + omega_z) * dt) * pz_1 + (omega_z * gamma * c ** 2 * dt) * pz_integral - \
    #      (c ** 2 * dt/dz) * correlate(vz_1, c1stOrd.T, mode='constant')
    #
    # px[z_f, x_f] += a_x * dt * source_integral / 2
    # pz[z_f, x_f] += a_x * dt * source_integral / 2

    # # without pml
    # p_1, p_2 = p, p_1
    # p = 2 * p_1 - p_2 + (dt * c[:, :] / dx) ** 2 * correlate(p_1, c2ndOrd, mode='constant')
    # p[Nz // 2, Nx // 2] += f[k]

    # Signal Acquisition
    for i in range(n_sensors):
        sensors[i, k] = px[sensors_pos[0, i], sensors_pos[1, i]] + pz[sensors_pos[0, i], sensors_pos[1, i]]

    # debugging mid-sim
    if k > 800:
        print(0)


    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt-1} - Math Time: {math_time - iteration_start} s")
    # print(np.shares_memory(u, uu))

    # Exhibition Update - QT
    # x = np.concatenate(((px+pz), p), 1)
    # window.imv.setImage(x.T, levels=[-0.1, 0.1])
    window.imv.setImage((px + pz).T, levels=[-5e-7, 5e-7])
    # window.imv.setImage((px + pz).T, levels=[np.min(px + pz), np.max(px + pz)])
    # window.imv.setImage((px + pz).T, levels=[0, 255])
    App.processEvents()

App.exit()

end_time = time.time()
total_time = end_time - start_time

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

## Plot of the recordings
## Signal Analyses - Single Emissor
fig_sig = plt.figure(2)
ax_s = fig_sig.add_subplot(111)
fig_sig.subplots_adjust(left=0.25, bottom=0.25)
plt.title('PV_PML')
element0 = 0
emissor0 = 0
delta_e = 1
im1 = ax_s.plot(t, sensors[element0]/np.linalg.norm(sensors[element0]), label=f'sim_e_r{element0}')
ax_s.legend()
#ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axelement = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
selement_sim = Slider(axelement, 'Receptor_sim', 0, n_sensors-1, valinit=element0, valstep=delta_e)
#axemissor_c = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
#semissor_c = Slider(axemissor_c, 'Emissor_sim', 0, n_es-1, valinit=emissor0, valstep=delta_e)

def update_rec_vs_data(val):
    element_sim = selement_sim.val
    #emissor_sim = semissor_c.val
    ax_s.cla()
    ax_s.plot(t, sensors[element_sim]/np.linalg.norm(sensors[element_sim]),
              label=f'sim_e_r{element_sim}')
    ax_s.legend()
    fig_sig.canvas.draw_idle()


selement_sim.on_changed(update_rec_vs_data)
#semissor_c.on_changed(update_rec_vs_data)

plt.show()

print(0)
