#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com CPML
# Komastitsch fortran 90 code translation
# Felipe Derewlany Gutierrez


from PyQt5.QtWidgets import *
import numpy as np
from scipy import signal
from time import perf_counter
import math
import matplotlib.pyplot as plt
from matplotlib import use
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# use('TkAgg')

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
        self.setWindowTitle(f"{NY}x{NX} Grid x {NSTEP} iterations - dx = {DELTAX} m x dy = {DELTAY} m x dt = {DELTAT} s")

        # setting geometry
        # self.setGeometry(200, 50, 1600, 800)
        self.setGeometry(200, 50, 1000, 1000)

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


## Simulation Parameters
# Flags to add PML layers to the edges of the grid
USE_PML_XMIN = True
USE_PML_XMAX = True
USE_PML_YMIN = True
USE_PML_YMAX = True

# number of points
NX = 801
NY = 801

# size of grid cell
DELTAX = 1.5  # [m? km?]
DELTAY = DELTAX

# Thickness of the PML layer in grid points
NPOINTS_PML = 10

# P-velocity and density
cp_unrelaxed = 2000.0  # [m/s] ??
density = 2000.0  #  [kg / m ** 3] ??

# Total number of time steps
NSTEP = 1500

# Time step in seconds
DELTAT = 5.2e-4  # [s]

# Parameters for the source
f0 = 35.0  # frequency? [Hz]
t0 = 1.20 / f0  # delay? [?]
factor = 1.0

# Source (in pressure)
xsource = 600.0
ysource = 600.0
ISOURCE = int(xsource / DELTAX) + 1
JSOURCE = int(ysource / DELTAY) + 1

# Receivers
NREC = 1
xdeb = 561.0   # First receiver x in meters
ydeb = 561.0   # First receiver y in meters
xfin = 561.0   # Last receiver x in meters
yfin = 561.0   # Last receiver y in meters


# Zero
ZERO = 0.0

# Large value for maximum
HUGEVAL = 1.0e30

# Threshold above which we consider that the code became unstable
STABILITY_THRESHOLD = 1.0e25

# Main arrays
pressure_past = np.zeros((NY, NX))
pressure_present = np.zeros((NY, NX))
pressure_future = np.zeros((NY, NX))
pressure_xx = np.zeros((NY, NX))
pressure_yy = np.zeros((NY, NX))
dpressurexx_dx = np.zeros((NY, NX))
dpressureyy_dy = np.zeros((NY, NX))
kappa_unrelaxed = np.zeros((NY, NX))
rho = np.zeros((NY, NX))
Kronecker_source = np.zeros((NY, NX))

# To interpolate material parameters or velocity at the right location in the staggered grid cell
rho_half_x = np.zeros((NY, NX))
rho_half_y = np.zeros((NY, NX))

# Power to compute d0 profile
NPOWER = 2.0

# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = 1.0
ALPHA_MAX_PML = 2.0 * math.pi * (f0 / 2)  # from Festa and Vilotte

# auxiliar
memory_dpressure_dx = np.zeros((NY, NX))
memory_dpressure_dy = np.zeros((NY, NX))
memory_dpressurexx_dx = np.zeros((NY, NX))
memory_dpressureyy_dy = np.zeros((NY, NX))

value_dpressure_dx = np.zeros((NY, NX))
value_dpressure_dy = np.zeros((NY, NX))
value_dpressurexx_dx = np.zeros((NY, NX))
value_dpressureyy_dy = np.zeros((NY, NX))

# Arrays for the damping profiles
d_x = np.zeros((NY, NX))
d_x_half = np.zeros((NY, NX))
K_x = np.zeros((NY, NX))
K_x_half = np.zeros((NY, NX))
alpha_x = np.zeros((NY, NX))
alpha_x_half = np.zeros((NY, NX))
a_x = np.zeros((NY, NX))
a_x_half = np.zeros((NY, NX))
b_x = np.zeros((NY, NX))
b_x_half = np.zeros((NY, NX))

d_y = np.zeros((NY, NX))
d_y_half = np.zeros((NY, NX))
K_y = np.zeros((NY, NX))
K_y_half = np.zeros((NY, NX))
alpha_y = np.zeros((NY, NX))
alpha_y_half = np.zeros((NY, NX))
a_y = np.zeros((NY, NX))
a_y_half = np.zeros((NY, NX))
b_y = np.zeros((NY, NX))
b_y_half = np.zeros((NY, NX))

thickness_PML_x = 0.0
thickness_PML_y = 0.0
xoriginleft = 0.0
xoriginright = 0.0
xoriginbottom = 0.0
xorigintop = 0.0
Rcoef = 0.0
d0_x = 0.0
d0_y = 0.0
xval = 0.0
yval = 0.0

# for source
a = 0.0
t = 0.0
source_term = 0.0

# for receivers
xspacerec = 0.0
yspacerec = 0.0
distval = 0.0
dist = 0.0
ix_rec = np.zeros(NREC, dtype='int')
iy_rec = np.zeros(NREC, dtype='int')
xrec = np.zeros(NREC)
yrec = np.zeros(NREC)
myNREC = NREC

# for seismograms
sispressure = np.zeros((NREC, NSTEP))

i = 0
j = 0
it = 0
irec = 0

Courant_number = 0.0
pressurenorm = 0.0

# program starts here
print("2D acoustic finite-difference code in pressure formulation with C-PML\n")
print(f"NX = {NX}\nNY = {NY}\n\n")
print(f"size of the model along X = {(NX-1)*DELTAX}")
print(f"size of the model along Y = {(NY-1)*DELTAY}\n")
print(f"Total number of grid points = {NX * NY}\n")

# define profile of absorption in PML region
#thickness of PML layers in meters
thickness_PML_x = NPOINTS_PML * DELTAX
thickness_PML_y = NPOINTS_PML * DELTAY

# reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
Rcoef = 0.001

# check that power is okay
if NPOWER < 1:
    raise ValueError('NPOWER must be greater than 1')

# compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
d0_x = -(NPOWER + 1) * cp_unrelaxed * math.log(Rcoef) / (2.0 * thickness_PML_x)
d0_y = -(NPOWER + 1) * cp_unrelaxed * math.log(Rcoef) / (2.0 * thickness_PML_y)

# d_x[:] = np.zeros(NX)
# d_x_half[:] = np.zeros(NX)
# K_x[:] = np.ones(NX)
# K_x_half[:] = np.ones(NX)
# alpha_x[:] = np.zeros(NX)
# alpha_x_half[:] = np.zeros(NX)
# a_x[:] = np.zeros(NX)
# a_x_half[:] = np.zeros(NX)
#
# d_y[:] = np.zeros(NY)
# d_y_half[:] = np.zeros(NY)
# K_y[:] = np.ones(NY)
# K_y_half[:] = np.ones(NY)
# alpha_y[:] = np.zeros(NY)
# alpha_y_half[:] = np.zeros(NY)
# a_y[:] = np.zeros(NY)
# a_y_half[:] = np.zeros(NY)

## damping in the X direction
# origin of the PML layer (position of right edge minus thickness, in meters)
xoriginleft = thickness_PML_x
xoriginright = (NX - 1) * DELTAX - thickness_PML_x

# dampening profile in X direction at the grid points
i = np.arange(NX)
xval = DELTAX * i
abscissa_in_PML_left = xoriginleft - xval
abscissa_in_PML_right = xval - xoriginright
abscissa_in_PML_mask_left = np.where(abscissa_in_PML_left < 0.0, False, True)
abscissa_in_PML_mask_right = np.where(abscissa_in_PML_right < 0.0, False, True)
abscissa_in_PML_mask = np.logical_or(abscissa_in_PML_mask_left, abscissa_in_PML_mask_right)
abscissa_in_PML = np.zeros(NX)
abscissa_in_PML[abscissa_in_PML_mask_left] = abscissa_in_PML_left[abscissa_in_PML_mask_left]
abscissa_in_PML[abscissa_in_PML_mask_right] = abscissa_in_PML_right[abscissa_in_PML_mask_right]
abscissa_in_PML_x = np.zeros((NY, NX))
abscissa_in_PML_x[:, :] = abscissa_in_PML
abscissa_normalized = abscissa_in_PML_x / thickness_PML_x
d_x[:, :] = d0_x * abscissa_normalized ** NPOWER
K_x[:, :] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_x[:, :] = ALPHA_MAX_PML * (1.0 - np.where(abscissa_in_PML_mask, abscissa_normalized, 1.0))

# dampening profile in X direction at half the grid points
abscissa_in_PML_left = xoriginleft - (xval + DELTAX / 2.0)
abscissa_in_PML_right = (xval + DELTAX/2.0) - xoriginright
abscissa_in_PML_mask_left = np.where(abscissa_in_PML_left < 0.0, False, True)
abscissa_in_PML_mask_right = np.where(abscissa_in_PML_right < 0.0, False, True)
abscissa_in_PML_mask = np.logical_or(abscissa_in_PML_mask_left, abscissa_in_PML_mask_right)
abscissa_in_PML = np.zeros(NX)
abscissa_in_PML[abscissa_in_PML_mask_left] = abscissa_in_PML_left[abscissa_in_PML_mask_left]
abscissa_in_PML[abscissa_in_PML_mask_right] = abscissa_in_PML_right[abscissa_in_PML_mask_right]
abscissa_in_PML_x_half = np.zeros((NY, NX))
abscissa_in_PML_x_half[:, :] = abscissa_in_PML
abscissa_normalized = abscissa_in_PML_x_half / thickness_PML_x
d_x_half[:, :] = d0_x * abscissa_normalized ** NPOWER
K_x_half[:, :] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_x_half[:, :] = ALPHA_MAX_PML * (1.0 - np.where(abscissa_in_PML_mask, abscissa_normalized, 1.0))

# just in case, for -5 at the end
# i = np.where(alpha_x < 0)
# alpha_x[i] = 0
# i = np.where(alpha_x_half < 0)
# alpha_x_half[i] = 0

b_x = np.exp(- (d_x / K_x + alpha_x) * DELTAT)
b_x_half = np.exp(- (d_x_half / K_x_half + alpha_x_half) * DELTAT)

# to avoid division by zero outside tha PML
i = np.where(d_x > 1e-6)
a_x[i] = d_x[i] * (b_x[i] - 1.0) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
i = np.where(d_x_half > 1e-6)
a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))


## damping in the Y direction
# origin of the PML layer (position of right edge minus thickness, in meters)
yoriginbottom = thickness_PML_y
yorigintop = (NY - 1) * DELTAY - thickness_PML_y

# dampening profile in Y direction at the grid points
j = np.arange(NY)
yval = DELTAY * j
abscissa_in_PML_bottom = yoriginbottom - yval
abscissa_in_PML_top = yval - yorigintop
abscissa_in_PML_mask_bottom = np.where(abscissa_in_PML_bottom < 0.0, False, True)
abscissa_in_PML_mask_top = np.where(abscissa_in_PML_top < 0.0, False, True)
abscissa_in_PML_mask = np.logical_or(abscissa_in_PML_mask_bottom, abscissa_in_PML_mask_top)
abscissa_in_PML = np.zeros(NY)
abscissa_in_PML[abscissa_in_PML_mask_bottom] = abscissa_in_PML_bottom[abscissa_in_PML_mask_bottom]
abscissa_in_PML[abscissa_in_PML_mask_top] = abscissa_in_PML_top[abscissa_in_PML_mask_top]
abscissa_in_PML_y = np.zeros((NY, NX))
abscissa_in_PML_y[:, :] = np.expand_dims(abscissa_in_PML, axis=1)
abscissa_normalized = abscissa_in_PML_y / thickness_PML_y
d_y[:, :] = d0_y * abscissa_normalized ** NPOWER
K_y[:, :] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_y[:, :] = ALPHA_MAX_PML * (1.0 - np.where(abscissa_in_PML_mask, abscissa_normalized.T, 1.0)).T

# dampening profile in X direction at half the grid points
abscissa_in_PML_bottom = yoriginbottom - (yval + DELTAY / 2.0)
abscissa_in_PML_top = (yval + DELTAX/2.0) - yorigintop
abscissa_in_PML_mask_bottom = np.where(abscissa_in_PML_bottom < 0.0, False, True)
abscissa_in_PML_mask_top = np.where(abscissa_in_PML_top < 0.0, False, True)
abscissa_in_PML_mask = np.logical_or(abscissa_in_PML_mask_bottom, abscissa_in_PML_mask_top)
abscissa_in_PML = np.zeros(NY)
abscissa_in_PML[abscissa_in_PML_mask_bottom] = abscissa_in_PML_bottom[abscissa_in_PML_mask_bottom]
abscissa_in_PML[abscissa_in_PML_mask_top] = abscissa_in_PML_top[abscissa_in_PML_mask_top]
abscissa_in_PML_y_half = np.zeros((NY, NX))
abscissa_in_PML_y_half[:, :] = np.expand_dims(abscissa_in_PML, axis=1)
abscissa_normalized = abscissa_in_PML_y_half / thickness_PML_y
d_y_half[:, :] = d0_y * abscissa_normalized ** NPOWER
K_y_half[:, :] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_y_half[:, :] = ALPHA_MAX_PML * (1.0 - np.where(abscissa_in_PML_mask, abscissa_normalized.T, 1.0)).T

b_y = np.exp(- (d_y / K_y + alpha_y) * DELTAT)
b_y_half = np.exp(- (d_y_half / K_y_half + alpha_y_half) * DELTAT)

# to avoid division by zero outside tha PML
j = np.where(d_y > 1e-6)
a_y[j] = d_y[j] * (b_y[j] - 1.0) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
j = np.where(d_y_half > 1e-6)
a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))

# # To compute stiffness (Lame parameter) and density
# for j in range(0, NY):
#     for i in range(0, NX):
#         kappa_unrelaxed[j, i] = density * cp_unrelaxed ** 2.0
#         rho[j, i] = density

# # Compute the stiffness (Lame parameter) and density
kappa_unrelaxed = density * cp_unrelaxed ** 2 * np.ones((NY, NX))
rho = density * np.ones((NY, NX))
rho_half_x[:, :-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
rho_half_x[:, NX-1] = rho_half_x[:, NX-2]
rho_half_y[:-1, :] = 0.5 * (rho[1:, :] + rho[:-1, :])
rho_half_y[NY-1, :] = rho_half_y[NY-2, :]

# source position
print(f"Position of the source: ")
print(f"x = {xsource}")
print(f"y = {ysource}\n")

# define source location
# Kronecker_source = np.zeros((NY, NX))
Kronecker_source[JSOURCE, ISOURCE] = 1

# define location of receivers
print(f"There are {NREC} receivers")

if NREC > 1:
    # this is to avoid a warning with GNU gfortran at compile time about division by zero when NREC = 1
    myNREC = NREC
    xspacerec = (xfin-xdeb) / (myNREC-1)
    yspacerec = (yfin-ydeb) / (myNREC-1)
else:
    xspacerec = 0
    yspacerec = 0

for irec in range(0, NREC):
    xrec[irec] = xdeb + (irec) * xspacerec
    yrec[irec] = ydeb + (irec) * yspacerec

# find closest grid point for each receiver
for irec in range(0, NREC):
    dist = HUGEVAL
    for j in range(0, NY):
        for i in range(0, NX):
            distval = np.sqrt((DELTAX*i - xrec[irec])**2 + (DELTAY*j - yrec[irec])**2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j

    print(f"Receiver {irec}:")
    print(f"x_target, y_target = {xrec[irec]}, {yrec[irec]}")
    print(f"Closest grid point at distance: {dist} in")
    print(f"i, j = {ix_rec}, {iy_rec}")

# Check the Courant stability condition for the explicit time scheme
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
Courant_number = cp_unrelaxed * DELTAT * np.sqrt(1.0/DELTAX ** 2 + 1.0/DELTAY ** 2)
print(f"Courant number is {Courant_number}")
if Courant_number > 1:
    print("time step is too large, simulation will be unstable")
    exit(1)


## Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()

# Start timer for simulation
start_time = perf_counter()

# beginning of time loop
# Main loop
# plt.figure()
for it in range(1, NSTEP):
    # Compute the first spatial derivatives divided by density
    # for j in range(0, NY):
    #     for i in range(0, NX-1):
    #         value_dpressure_dx = (pressure_present[j, i+1] - pressure_present[j, i]) / DELTAX
    #
    #         memory_dpressure_dx[j, i] = b_x_half[i] * memory_dpressure_dx[j, i] + a_x_half[i] * value_dpressure_dx
    #
    #         value_dpressure_dx = value_dpressure_dx / K_x_half[i] + memory_dpressure_dx[j, i]
    #
    #         rho_half_x = 0.5 * (rho[j, i+1] + rho[j, i])
    #         pressure_xx[j, i] = value_dpressure_dx / rho_half_x
    #
    # for j in range(0, NY-1):
    #     for i in range(0, NX):
    #         value_dpressure_dy = (pressure_present[j+1, i] - pressure_present[j, i]) / DELTAY
    #
    #         memory_dpressure_dy[j, i] = b_y_half[j] * memory_dpressure_dy[j, i] + a_y_half[j] * value_dpressure_dy
    #
    #         value_dpressure_dy = value_dpressure_dy / K_y_half[j] + memory_dpressure_dy[j, i]
    #
    #         rho_half_y = 0.5 * (rho[j+1, i] + rho[j, i])
    #         pressure_yy[j, i] = value_dpressure_dy / rho_half_y
    #
    # Compute the second spatial derivatives
    #     for j in range(0, NY):
    #         for i in range(1, NX):
    #             value_dpressurexx_dx = (pressure_xx[j, i] - pressure_xx[i-1, j]) / DELTAX
    #
    #             memory_dpressurexx_dx[j, i] = b_x[i] * memory_dpressurexx_dx[j, i] + a_x[i] * value_dpressurexx_dx
    #
    #             value_dpressurexx_dx = value_dpressurexx_dx / K_x[i] + memory_dpressurexx_dx[j, i]
    #
    #             dpressurexx_dx[j, i] = value_dpressurexx_dx
    #
    #     for j in range(1, NY):
    #         for i in range(0, NX):
    #             value_dpressureyy_dy = (pressure_yy[j, i] - pressure_yy[j-1, i]) / DELTAY
    #
    #             memory_dpressureyy_dy[j, i] = b_y[j] * memory_dpressureyy_dy[j, i] + a_y[j] * value_dpressureyy_dy
    #
    #             value_dpressureyy_dy = value_dpressureyy_dy / K_y[j] + memory_dpressureyy_dy[j, i]
    #
    #             dpressureyy_dy[j, i] = value_dpressureyy_dy


    # Compute the first spatial derivatives divided by density
    value_dpressure_dx[:, :-1] = (pressure_present[:, 1:] - pressure_present[:, :-1]) / DELTAX
    memory_dpressure_dx = b_x_half * memory_dpressure_dx + a_x_half * value_dpressure_dx
    value_dpressure_dx = value_dpressure_dx / K_x_half + memory_dpressure_dx
    pressure_xx = value_dpressure_dx / rho_half_x

    value_dpressure_dy[:-1, :] = (pressure_present[1:, :] - pressure_present[:-1, :]) / DELTAY
    memory_dpressure_dy = b_y_half * memory_dpressure_dy + a_y_half * value_dpressure_dy
    value_dpressure_dy = value_dpressure_dy / K_y_half + memory_dpressure_dy
    pressure_yy = value_dpressure_dy / rho_half_y

    # Compute the second spatial derivatives
    value_dpressurexx_dx[:, 1:] = (pressure_xx[:, 1:] - pressure_xx[:, :-1]) / DELTAX
    memory_dpressurexx_dx = b_x * memory_dpressurexx_dx + a_x * value_dpressurexx_dx
    value_dpressurexx_dx = value_dpressurexx_dx / K_x + memory_dpressurexx_dx
    dpressurexx_dx = value_dpressurexx_dx

    value_dpressureyy_dy[1:, :] = (pressure_yy[1:, :] - pressure_yy[:-1, :]) / DELTAY
    memory_dpressureyy_dy = b_y * memory_dpressureyy_dy + a_y * value_dpressureyy_dy
    value_dpressureyy_dy = value_dpressureyy_dy / K_y + memory_dpressureyy_dy
    dpressureyy_dy = value_dpressureyy_dy


    # add the source (pressure located at a given grid point)
    a = math.pi ** 2 * f0 ** 2
    t = (it - 1) * DELTAT

    # Gaussian
    # source_term = - factor * np.exp(-a * (t-t0) ** 2) / (2.0 * a)

    # first derivative of a Gaussian
    # source_term = factor * (t - t0) * np.exp(-a * (t-t0) ** 2)

    # Ricker source time function (second derivative of a Gaussian)
    source_term = factor * (1.0 - 2.0 * a * (t-t0) ** 2) * np.exp(-a * (t-t0) ** 2)

    # apply the time evolution scheme
    # we apply it everywhere, including at some points on the edges of the domain that have not be calculated above,
    # which is of course wrong (or more precisely undefined), but this does not matter because these values
    # will be erased by the Dirichlet conditions set on these edges below

    pressure_future = 2.0 * pressure_present - pressure_past + \
        DELTAT ** 2 * ((dpressurexx_dx + dpressureyy_dy) * kappa_unrelaxed + \
                              4.0 * math.pi * cp_unrelaxed ** 2 * source_term * Kronecker_source)

    ## apply Dirichlet conditions at the bottom of the C-PML layers
    ## which is the right condition to implement in order for C-PML to remain stable at long times
    # Dirichlet condition for pressure on the left boundary
    pressure_future[:, 0] = 0

    # Dirichlet condition for pressure on the right boundary
    pressure_future[:, NX-1] = 0

    # Dirichlet condition for pressure on the bottom boundary
    pressure_future[0, :] = 0

    # Dirichlet condition for pressure on the top boundary
    pressure_future[NY-1, :] = 0

    # Store seismograms
    for irec in range(0, NREC):
        sispressure[irec, it] = pressure_future[iy_rec[irec], ix_rec[irec]]

    # if (it > 400) and (it % 15) == 0:
    #     plt.plot(pressure_future[400, :])
    #     plt.show()

    # print maximum of pressure and of norm of velocity
    pressurenorm = np.max(np.abs(pressure_future))
    print(f"Time step {it} out of {NSTEP}")
    print(f"Time: {(it - 1) * DELTAT} seconds")
    print(f"Max absolute value of pressure = {pressurenorm}")

    # check stability of the code, exit if unstable
    if pressurenorm > STABILITY_THRESHOLD:
        print("code became unstable and blew up")
        exit(2)

    window.imv.setImage(pressure_future.T, levels=[-1.0, 1.0])
    # window.imv.setImage(np.log(pressure_future.T), levels=[-0.5, 0.5])
    App.processEvents()

    # move new values to old values (the present becomes the past, the future becomes the present)
    pressure_past = pressure_present
    pressure_present = pressure_future


App.exit()
end_time = perf_counter()
total_time = end_time - start_time

# End of the main loop
print("Simulation finished.")
print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")


print("END")



