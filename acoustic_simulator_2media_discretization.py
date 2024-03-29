#! /usr/bin/env python3
# Port do algoritmo do Prof Dr Pipa
# Utilizado para os testes de dispersão numérica
# Controlando os parâmetros de discretização diretamente

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import correlate
import time
import math
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


def speed_measuring(z1, z2, pressure_at_z1, pressure_at_z2, dh, medium_speed, mode=0):

    '''
    # Wave Speed Estimation [pixel/iteration]
    delta_pixels1 = z2 - z1
    u_delta_iterations1 = np.where(pressure_at_z2 == max(pressure_at_z2))[0] - \
                          np.where(pressure_at_z1 == max(pressure_at_z1))[0]
    u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

    # Wave Speed Estimation [m/s]
    space1 = (z2 - z1) * dh
    u_propagation_time1 = t[np.where(pressure_at_z2 == max(pressure_at_z2))[0]] - \
                          t[np.where(pressure_at_z1 == max(pressure_at_z1))[0]]
    u_wave_speed1 = space1 / u_propagation_time1
    '''

    #print(f"\nMedia {medium_speed}")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1} m/s")

    if mode == 0:
        # Wave Speed Estimation [pixel/iteration]
        delta_pixels1 = z2 - z1

        envelope1 = np.abs(signal.hilbert(pressure_at_z1))
        envelope2 = np.abs(signal.hilbert(pressure_at_z2))

        u_delta_iterations1 = np.where(envelope2 == max(envelope2))[0] - \
                              np.where(envelope1 == max(envelope1))[0]
        u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

        print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
        return u_wave_speed1pi
    elif mode == 1:
        # Wave Speed Estimation [m/s]
        space1 = (z2 - z1) * dh

        envelope1 = np.abs(signal.hilbert(pressure_at_z1))
        envelope2 = np.abs(signal.hilbert(pressure_at_z2))

        u_propagation_time1 = t[np.where(envelope2 == max(envelope2))[0]] - \
                              t[np.where(envelope1 == max(envelope1))[0]]
        u_wave_speed1 = space1 / u_propagation_time1

        print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1} m/s")
        return u_wave_speed1
    else:
        return -1


# Parameters
T = 0.000008  # [s]
Lx = 0.02  # [m]
Lz = 0.02  # [m]
dt = 3.125e-9  # [m/pixel]
dx = 6.25e-5  # [m/pixel]
dz = dx
Nt = math.ceil(T / dt)
Nx = math.ceil(Lx / dx)
Nz = math.ceil(Lz / dz)

# Souce
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 5e6  # [Hz]
delay = 1e-6
bandwidth = 0.6
f = signal.gausspulse(t - delay, frequency, bandwidth)

print(f"{Nx}x{Nz} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

# Speed
#soundspeed = 1000  # [m/s]
#soundspeed1 = 1481  # [m/s]
soundspeed1 = 2500  # [m/s]
#soundspeed = 3000  # [m/s]
soundspeed2 = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

# Propagation media definition
media = np.zeros((Nz, Nx))
interface = round(Nz/3)
media[0:interface, 0:Nx] += soundspeed1 * np.ones((interface, Nx))  # Água do mar
media[interface:Nz, 0:Nx] += soundspeed2 * np.ones(((Nz-interface), Nx))  # Aço

c = media / ad

CFL1 = soundspeed1 * dt / dx
CFL2 = soundspeed2 * dt / dx

print(f"CFL condition = {CFL1}")
print(f"CFL condition = {CFL2}")

# Pressure Fields
u = np.zeros((Nz, Nx))
u_1 = np.zeros((Nz, Nx))
u_2 = np.zeros((Nz, Nx))

u_at_transducer = np.zeros(Nt)
u_at_poi1 = np.zeros(Nt)
u_at_poi2 = np.zeros(Nt)
u_at_poi3 = np.zeros(Nt)
u_at_poi4 = np.zeros(Nt)
u_at_poi5 = np.zeros(Nt)
u_at_poi6 = np.zeros(Nt)
u_at_poi7 = np.zeros(Nt)
u_at_poi8 = np.zeros(Nt)
u_at_poi9 = np.zeros(Nt)

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
# print(str(np.size(coeff, 0)) + " x " + str((np.size(coeff, 1))))

# Simulation Parameters
delta = 3e-3
z_f = round(Nz * 0)  # Transductor y coordinate
x_f = round(Nx / 2)
x_pois = x_f
z_poi1 = round(Nx * 0.2)
z_poi2 = round(Nx * 0.25)
z_poi3 = round(Nx * 0.3)
z_poi4 = round(Nx * 0.35)
z_poi5 = round(Nx * 0.65)
z_poi6 = round(Nx * 0.7)
z_poi7 = round(Nx * 0.75)
z_poi8 = round(Nx * 0.8)
z_poi9 = round(Nx * 0.85)

measured_speed1_pi = []
measured_speed1_ms = []
measured_speed2_pi = []
measured_speed2_ms = []

# Exhibition setup
sfmt = pg.QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
pg.QtGui.QSurfaceFormat.setDefaultFormat(sfmt)

app = pg.QtGui.QApplication([])
riw = pg.widgets.RawImageWidget.RawImageWidget()
riw.show()

start_time = time.time()

x = np.zeros((Nz, 2 * Nx))

for k in range(4, Nt):
    iteration_start = time.time()

    u_0 = u
    u_2 = u_1
    u_1 = u_0

    if deriv_accuracy < 16:
        lap = correlate(u_1[:, :], coeff)
    else:
        lap = signal.fftconvolve(u_1[:, :], coeff, mode='same')


    #u = (2 * u_1[:, :] - (1 - delta / 2) * u_2[:, :] + (c[:, :] ** 2) * lap) / (1 + delta / 2)
    u = 2 * u_1[:, :] - u_2[:, :] + (c[:, :] ** 2) * lap

    u[z_f, x_f] += f[k]

    # Speed measuring
    u_at_transducer[k] = u[z_f, x_f]
    u_at_poi1[k] = u[z_poi1, x_f]
    u_at_poi2[k] = u[z_poi2, x_f]
    u_at_poi3[k] = u[z_poi3, x_f]
    u_at_poi4[k] = u[z_poi4, x_f]
    u_at_poi5[k] = u[z_poi5, x_f]
    u_at_poi6[k] = u[z_poi6, x_f]
    u_at_poi7[k] = u[z_poi7, x_f]
    u_at_poi8[k] = u[z_poi8, x_f]
    u_at_poi9[k] = u[z_poi9, x_f]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")

    # Exhibition - QT
    riw.setImage(u.T, levels=[-0.1, 0.1])
    app.processEvents()

app.exit()

end_time = time.time()
total_time = end_time - start_time


print(f"\n\n{Nx}x{Nz} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")

print("\nMeasured Soundspeed for media 1 [pixels/timestep]")
measured_speed1_pi.append(speed_measuring(z_f, z_poi1, u_at_transducer, u_at_poi1, dz, soundspeed1, mode=0))
measured_speed1_pi.append(speed_measuring(z_f, z_poi2, u_at_transducer, u_at_poi2, dz, soundspeed1, mode=0))
measured_speed1_pi.append(speed_measuring(z_f, z_poi3, u_at_transducer, u_at_poi3, dz, soundspeed1, mode=0))

print("\nMeasured Soundspeed for media 2 [pixels/timestep]")
measured_speed2_pi.append(speed_measuring(z_poi4, z_poi5, u_at_poi4, u_at_poi5, dz, soundspeed2, mode=0))
measured_speed2_pi.append(speed_measuring(z_poi4, z_poi6, u_at_poi4, u_at_poi6, dz, soundspeed2, mode=0))
measured_speed2_pi.append(speed_measuring(z_poi4, z_poi7, u_at_poi4, u_at_poi7, dz, soundspeed2, mode=0))
measured_speed2_pi.append(speed_measuring(z_poi4, z_poi8, u_at_poi4, u_at_poi8, dz, soundspeed2, mode=0))
measured_speed2_pi.append(speed_measuring(z_poi4, z_poi9, u_at_poi4, u_at_poi9, dz, soundspeed2, mode=0))

print("\nMeasured Soundspeed for media 1 [m/s]")
measured_speed1_ms.append(speed_measuring(z_f, z_poi1, u_at_transducer, u_at_poi1, dz, soundspeed1, mode=1))
measured_speed1_ms.append(speed_measuring(z_f, z_poi2, u_at_transducer, u_at_poi2, dz, soundspeed1, mode=1))
measured_speed1_ms.append(speed_measuring(z_f, z_poi3, u_at_transducer, u_at_poi3, dz, soundspeed1, mode=1))

print("\nMeasured Soundspeed for media 2 [m/s]")
measured_speed2_ms.append(speed_measuring(z_poi4, z_poi5, u_at_poi4, u_at_poi5, dz, soundspeed2, mode=1))
measured_speed2_ms.append(speed_measuring(z_poi4, z_poi6, u_at_poi4, u_at_poi6, dz, soundspeed2, mode=1))
measured_speed2_ms.append(speed_measuring(z_poi4, z_poi7, u_at_poi4, u_at_poi7, dz, soundspeed2, mode=1))
measured_speed2_ms.append(speed_measuring(z_poi4, z_poi8, u_at_poi4, u_at_poi8, dz, soundspeed2, mode=1))
measured_speed2_ms.append(speed_measuring(z_poi4, z_poi9, u_at_poi4, u_at_poi9, dz, soundspeed2, mode=1))

print(f"\nMedia {soundspeed1}")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_pi)} pixels/timestep")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_ms)} m/s")

print(f"\nMedia {soundspeed2}")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed2_pi)} pixels/timestep")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed2_ms)} m/s")


print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

plt.ioff()
plt.figure(1)
plt.plot(t, f/np.linalg.norm(f), 'b', label='Fonte')
plt.plot(t, u_at_transducer/np.linalg.norm(u_at_transducer), 'r', label='Medido')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title("Pressão no pixel do transdutor")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(t, u_at_transducer/np.linalg.norm(u_at_transducer), 'r', label='Medido na fonte')
plt.plot(t, u_at_poi1/np.linalg.norm(u_at_poi1))
plt.plot(t, u_at_poi2/np.linalg.norm(u_at_poi2))
plt.plot(t, u_at_poi3/np.linalg.norm(u_at_poi3))
plt.plot(t, u_at_poi4/np.linalg.norm(u_at_poi4))
plt.plot(t, u_at_poi5/np.linalg.norm(u_at_poi5))
plt.plot(t, u_at_poi6/np.linalg.norm(u_at_poi6))
plt.plot(t, u_at_poi7/np.linalg.norm(u_at_poi7))
plt.plot(t, u_at_poi8/np.linalg.norm(u_at_poi8))
plt.plot(t, u_at_poi9/np.linalg.norm(u_at_poi9))
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel do transdutor vs nos pontos de interesse - Lap {deriv_accuracy}")
plt.legend()
plt.show()

envelopef = np.abs(signal.hilbert(u_at_transducer))
plt.figure(3)
plt.plot(t, u_at_transducer, 'b', label='Medido na fonte')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel do transdutor")
plt.legend()
plt.show()

envelopef = np.abs(signal.hilbert(u_at_poi9))
plt.figure(4)
plt.plot(t, u_at_poi9, 'b', label='Medido no ponto 9')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel poi9")
plt.legend()
plt.show()

plt.figure(5)
plt.plot(t, u_at_transducer/np.linalg.norm(u_at_transducer), 'b', label='Medido na fonte')
plt.plot(t, np.abs(signal.hilbert(u_at_transducer))/np.linalg.norm(u_at_transducer), 'g', label='Envelope')
plt.plot(t, u_at_poi3/np.linalg.norm(u_at_poi3), 'r', label='Medido no ponto 3')
plt.plot(t, np.abs(signal.hilbert(u_at_poi3))/np.linalg.norm(u_at_poi3), 'c', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Medição da velocidade meio {soundspeed1}")
plt.legend()
plt.show()

plt.figure(6)
plt.plot(t, u_at_poi4/np.linalg.norm(u_at_poi4), 'b', label='Medido no ponto 4')
plt.plot(t, np.abs(signal.hilbert(u_at_poi4))/np.linalg.norm(u_at_poi4), 'g', label='Envelope')
plt.plot(t, u_at_poi9/np.linalg.norm(u_at_poi9), 'r', label='Medido no ponto 9')
plt.plot(t, np.abs(signal.hilbert(u_at_poi9))/np.linalg.norm(u_at_poi9), 'c', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Medição da velocidade meio {soundspeed2}")
plt.legend()
plt.show()
