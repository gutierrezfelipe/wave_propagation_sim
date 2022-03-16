#! /usr/bin/env python3
# Port do algoritmo do Prof Pipa

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import correlate
import time
import math
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget


def speed_measuring(y1, y2, pressure_at_y1, pressure_at_y2, dh, medium_speed, mode=0):

    '''
    # Wave Speed Estimation [pixel/iteration]
    delta_pixels1 = y2 - y1
    u_delta_iterations1 = np.where(pressure_at_y2 == max(pressure_at_y2))[0] - \
                          np.where(pressure_at_y1 == max(pressure_at_y1))[0]
    u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

    # Wave Speed Estimation [m/s]
    space1 = (y2 - y1) * dh
    u_propagation_time1 = t[np.where(pressure_at_y2 == max(pressure_at_y2))[0]] - \
                          t[np.where(pressure_at_y1 == max(pressure_at_y1))[0]]
    u_wave_speed1 = space1 / u_propagation_time1
    '''

    #print(f"\nMedia {medium_speed}")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1} m/s")

    if mode == 0:
        # Wave Speed Estimation [pixel/iteration]
        delta_pixels1 = y2 - y1

        envelope1 = np.abs(signal.hilbert(pressure_at_y1))
        envelope2 = np.abs(signal.hilbert(pressure_at_y2))

        u_delta_iterations1 = np.where(envelope2 == max(envelope2))[0] - \
                              np.where(envelope1 == max(envelope1))[0]
        u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

        print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
        return u_wave_speed1pi
    elif mode == 1:
        # Wave Speed Estimation [m/s]
        space1 = (y2 - y1) * dh

        envelope1 = np.abs(signal.hilbert(pressure_at_y1))
        envelope2 = np.abs(signal.hilbert(pressure_at_y2))

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
Ly = 0.02  # [m]
dt = 5e-9  # [m/pixel]
dx = 10e-5  # [m/pixel]
dy = dx
Nt = math.ceil(T / dt)
Nx = math.ceil(Lx / dx)
Ny = math.ceil(Ly / dy)

print(f"{Nx}x{Ny} Grid x {Nt} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")

ad = math.sqrt((dx * dy) / (dt ** 2))  # Adimensionality constant

#soundspeed = 1000  # [m/s]
#soundspeed1 = 1481  # [m/s]
soundspeed = 2500  # [m/s]
#soundspeed2 = 3000  # [m/s]
#soundspeed2 = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

c = soundspeed / ad
#media = np.zeros((Ny, Nx))
#interface = round(Ny/3)
#media[0:interface, 0:Nx] += soundspeed1 * np.ones((interface, Nx))  # Água do mar
#media[interface:Ny, 0:Nx] += soundspeed2 * np.ones(((Ny-interface), Nx))  # Aço

#c = media / ad

#print(media)

CFL1 = soundspeed * dt / dx
#CFL2 = soundspeed2 * dt / dx

print(f"CFL condition = {CFL1}")
#print(f"CFL condition = {CFL2}")

u = np.zeros((Ny, Nx))
u_1 = np.zeros((Ny, Nx))
u_2 = np.zeros((Ny, Nx))
#u_3 = np.zeros((Ny, Nx))
#u_4 = np.zeros((Ny, Nx))

#v = np.zeros((Ny, Nx))
#v_1 = np.zeros((Ny, Nx))
#v_2 = np.zeros((Ny, Nx))

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
#u_at_poi10 = np.zeros(Nt)

#v_at_transducer = np.zeros(Nt)
#v_at_poi1 = np.zeros(Nt)
#v_at_poi2 = np.zeros(Nt)
#v_at_poi3 = np.zeros(Nt)

#fiber = np.zeros((Nt, Nx))

# nome = f"{Nx}x{Ny} x {Nt} - {dx}x{dy} x {dt}  - Pulso Gaussiano 50MHz - Laplaciano 2 x 20"

# Sources

# f1 = 1000
# f3 = 500

# sigma1 = 1e5
# sigma2 = 1e8
# sigma3 = 1e5

# t1 = 40e-4
# t2 = 5e-4
# t3 = 100e-4

t = np.linspace(0, T - dt, Nt)

frequency = 5e6  # [Hz]
delay = 1e-6
bandwidth = 0.6

f = signal.gausspulse(t - delay, frequency, bandwidth)
# s1 = -2*sigma*(t-t1).*exp(-sigma*(t-t1).^2)
# s1 = np.exp(-(t-t1)**2*sigma1)*np.cos(2*np.pi*f1*(t-t1))
# s2 = -2*sigma2*(t-t2)*np.exp(-sigma2*(t-t2)**2)
# s3 = np.exp(-(t-t3)**2*sigma1)*np.cos(2*np.pi*f3*(t-t3))
# s = exp(-sigma*(t-t0).^2).*cos(2*pi*f0*(t-t0))
# s1 = s1/max(s1)
# s2 = s2/max(s2)/2
# s3 = s3/max(s3)/5

#plt.ioff()
#plt.figure(1)
# plt.plot(t, s1, 'b', t, s2, 'r', t, s3, 'g')
#plt.plot(t, f, 'b', label='Fonte')
#plt.xlabel('Tempo [s]')
#plt.ylabel('Amplitude do sinal')
#plt.title("Sinal da fonte")
#plt.legend()
#plt.grid()
#plt.show()
# print("ok")

# Laplacian Kernels

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

#deriv_order = 2
#deriv_accuracy = 20
#deriv_n_coef = 2 * np.floor((deriv_order + 1) / 2).astype('int') - 1 + deriv_accuracy
#p = np.round((deriv_n_coef - 1) / 2).astype('int')
#A = np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[None].T
#b = np.zeros(2 * p + 1)
#b[deriv_order] = math.factorial(deriv_order)
#coeff2 = np.zeros((deriv_n_coef, deriv_n_coef))
# Solve system A*w = b
#coeff2[deriv_n_coef // 2, :] = np.linalg.solve(A, b)
#coeff2 += coeff2.T
# print(str(np.size(coeff2, 0)) + " x " + str((np.size(coeff2, 1))))


# Simulation

delta = 3e-3
y_fiber = round(Ny / 3)
# y_source = round(2*Ny/3)
y_f = round(Ny * 0)
x_f = round(Nx / 2)
# x_s1 = round(Nx/4)
# x_s2 = round(2*Nx/4)
# x_s3 = round(3*Nx/4)
x_pois = x_f
y_poi1 = round(Nx * 0.2)
#x_poi2 = x_f
y_poi2 = round(Nx * 0.25)
#x_poi3 = x_f
y_poi3 = round(Nx * 0.3)  # last sensor for 1st media
y_poi4 = round(Nx * 0.35)  # beginning of 2nd media
y_poi5 = round(Nx * 0.55)
y_poi6 = round(Nx * 0.6)
y_poi7 = round(Nx * 0.65)
y_poi8 = round(Nx * 0.7)
y_poi9 = round(Nx * 0.75)

measured_speed1_pi = []
measured_speed1_ms = []
measured_speed2_pi = []
measured_speed2_ms = []

# Writer Config
# metadata = dict(title=nome, author='AUSPEX')
# writer = matplotlib.animation.FFMpegWriter(fps=60, metadata=metadata)
# fig, ax = plt.subplots(figsize=(4, 4))
# im = ax.imshow(u[:, :], vmin=0, vmax=1)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
# im1 = ax1.imshow(u[:, :], vmin=0, vmax=1)
# im2 = ax2.imshow(v[:, :], vmin=0, vmax=1)
# ax1.imshow(meio1, alpha=0.1)
# ax1.imshow(checkpoint, alpha=0.5)

# plt.colorbar()
# plt.xlabel('x [mm]')
# plt.ylabel('z [mm]')

# Exhibition setup
sfmt = pg.QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
pg.QtGui.QSurfaceFormat.setDefaultFormat(sfmt)

app = pg.QtGui.QApplication([])
riw = pg.widgets.RawImageWidget.RawImageGLWidget()
riw.show()

start_time = time.time()

x = np.zeros((Ny, 2 * Nx))

# with writer.saving(fig, f"C:/AUSPEX/pipasim/{nome}.mp4", 400):
for k in range(4, Nt):
    iteration_start = time.time()

    #u_0 = u
    #u_2 = u_1
    #u_1 = u_0

    u_0 = u
    #u_4 = u_3
    #u_3 = u_2
    u_2 = u_1
    u_1 = u_0


    #v_0 = v
    #v_2 = v_1
    #v_1 = v_0

    if deriv_accuracy < 16:
        lap1 = correlate(u_1[:, :], coeff)
    else:
        lap1 = signal.fftconvolve(u_1[:, :], coeff, mode='same')

    #if deriv_accuracy < 16:
    #    lap2 = correlate(v_1[:, :], coeff)
    #else:
    #    lap2 = signal.fftconvolve(v_1[:, :], coeff, mode='same')

    u = (2 * u_1[:, :] - (1 - delta / 2) * u_2[:, :] + (c ** 2) * lap1) / (1 + delta / 2)
    #v = (2 * v_1[:, :] - (1 - delta / 2) * v_2[:, :] + (c ** 2) * lap2) / (1 + delta / 2)

    # Delta t 4
    # Backwards Difference
    #u = (104 * u_1[:, :] - 114 * u_2[:, :] + 56 * u_3[:, :] - 11 * u_4[:, :] + 12 * (c ** 2) * lap1) / 35
    # Central Difference - Correct
    #u = 16/12 * u_1[:, :] - 30/12 * u_2[:, :] + 16/12 * u_3[:, :] - 1/12 * u_4[:, :] - (c ** 2) * lap1

    u[y_f, x_f] += f[k]
    # u[y_source, x_s1] += s1[k]
    # u[y_source, x_s2] += s2[k]
    # u[y_source, x_s3] += s3[k]

    #v[y_f, x_f] += f[k]
    # v[y_source, x_s1] += s1[k]
    # v[y_source, x_s2] += s2[k]
    # v[y_source, x_s3] += s3[k]

    #fiber[k, :] = u[y_fiber, :].T

    #x[:, :Nx] = u
    #x[:, Nx:] = v

    # Speed measuring
    u_at_transducer[k] = u[y_f, x_f]
    u_at_poi1[k] = u[y_poi1, x_f]
    u_at_poi2[k] = u[y_poi2, x_f]
    u_at_poi3[k] = u[y_poi3, x_f]
    u_at_poi4[k] = u[y_poi4, x_f]
    u_at_poi5[k] = u[y_poi5, x_f]
    u_at_poi6[k] = u[y_poi6, x_f]
    u_at_poi7[k] = u[y_poi7, x_f]
    u_at_poi8[k] = u[y_poi8, x_f]
    u_at_poi9[k] = u[y_poi9, x_f]


    #v_at_transducer[k] = v[y_f, x_f]
    #v_at_poi1[k] = v[y_poi1, x_poi1]
    #v_at_poi2[k] = v[y_poi2, x_poi2]
    #v_at_poi3[k] = v[y_poi3, x_poi3]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")

    # Exhibition - video
    # uv = np.concatenate((u, v), axis=0)
    # im = ax.imshow(uv[:, :], vmin=0, vmax=1)
    # im.set_data(uv[:, :])
    # ax.set_title(f"u[{(k * dt):15.9f} s]")

    # im1.set_data(u[:, :])
    # ax1.set_title(f"u[{(k * dt):15.9f} s]")

    # im2.set_data(v[:, :])
    # ax2.set_title(f"v[{(k * dt):15.9f} s]")

    # writer.grab_frame()

    # print(f"Video Frame Time: {time.time() - math_time} s")

    # Exhibition - QT
    riw.setImage(u.T, levels=[-0.1, 0.1])
    app.processEvents()

app.exit()

end_time = time.time()
total_time = end_time - start_time

# plt.figure(2)
# plt.plot(t, fiber)
# plt.show()

#points_per_lambda = (soundspeed / (2 * frequency)) / dx
#points_per_period = (1 / (2 * frequency)) / dt

print(f"\n\n{Nx}x{Ny} Grid x {Nt} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")

#print(f"{points_per_lambda} points / lambda")
#print(f"{points_per_period} points / period")

print("\nMeasured Soundspeed for media 1 [pixels/timestep]")
measured_speed1_pi.append(speed_measuring(y_f, y_poi1, u_at_transducer, u_at_poi1, dy, soundspeed, mode=0))
measured_speed1_pi.append(speed_measuring(y_f, y_poi2, u_at_transducer, u_at_poi2, dy, soundspeed, mode=0))
measured_speed1_pi.append(speed_measuring(y_f, y_poi3, u_at_transducer, u_at_poi3, dy, soundspeed, mode=0))

#print("\nMeasured Soundspeed for media 2 [pixels/timestep]")
#measured_speed2_pi.append(speed_measuring(y_poi4, y_poi5, u_at_transducer, u_at_poi5, dy, soundspeed2, mode=0))
#measured_speed2_pi.append(speed_measuring(y_poi4, y_poi6, u_at_transducer, u_at_poi6, dy, soundspeed2, mode=0))
#measured_speed2_pi.append(speed_measuring(y_poi4, y_poi7, u_at_transducer, u_at_poi7, dy, soundspeed2, mode=0))
#measured_speed2_pi.append(speed_measuring(y_poi4, y_poi8, u_at_transducer, u_at_poi8, dy, soundspeed2, mode=0))
#measured_speed2_pi.append(speed_measuring(y_poi4, y_poi9, u_at_transducer, u_at_poi9, dy, soundspeed2, mode=0))

print("\nMeasured Soundspeed for media 1 [m/s]")
measured_speed1_ms.append(speed_measuring(y_f, y_poi1, u_at_transducer, u_at_poi1, dy, soundspeed, mode=1))
measured_speed1_ms.append(speed_measuring(y_f, y_poi2, u_at_transducer, u_at_poi2, dy, soundspeed, mode=1))
measured_speed1_ms.append(speed_measuring(y_f, y_poi3, u_at_transducer, u_at_poi3, dy, soundspeed, mode=1))

#print("\nMeasured Soundspeed for media 2 [m/s]")
#measured_speed2_ms.append(speed_measuring(y_poi4, y_poi6, u_at_transducer, u_at_poi6, dy, soundspeed2, mode=1))
#measured_speed2_ms.append(speed_measuring(y_poi4, y_poi7, u_at_transducer, u_at_poi7, dy, soundspeed2, mode=1))
#measured_speed2_ms.append(speed_measuring(y_poi4, y_poi8, u_at_transducer, u_at_poi8, dy, soundspeed2, mode=1))
#measured_speed2_ms.append(speed_measuring(y_poi4, y_poi9, u_at_transducer, u_at_poi9, dy, soundspeed2, mode=1))


print(f"\nMedia {soundspeed}")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_pi)} pixels/timestep")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_ms)} m/s")

#print(f"\nMedia {soundspeed2}")
#print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed2_pi)} pixels/timestep")
#print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed2_ms)} m/s")


print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

plt.ioff()
plt.figure(1)
plt.plot(t, f, 'b', label='Fonte')
plt.plot(t, u_at_transducer, 'r', label='Medido')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title("Pressão no pixel do transdutor")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(t, u_at_transducer, 'r', label='Medido na fonte')
plt.plot(t, u_at_poi1, label='Medido em outro pixel')
plt.plot(t, u_at_poi2)
plt.plot(t, u_at_poi3)
plt.plot(t, u_at_poi4)
plt.plot(t, u_at_poi5)
plt.plot(t, u_at_poi6)
plt.plot(t, u_at_poi7)
plt.plot(t, u_at_poi8)
plt.title(f"Pressão no pixel do transdutor vs nos pontos de interesse - Lap {deriv_accuracy}")
plt.legend()
plt.show()

envelopef = np.abs(signal.hilbert(u_at_transducer))
plt.figure(3)
plt.plot(t, u_at_transducer, 'b', label='Medido na fonte')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.title(f"Pressão no pixel do transdutor")
plt.legend
plt.show()

envelopef = np.abs(signal.hilbert(u_at_poi9))
plt.figure(4)
plt.plot(t, u_at_poi9, 'b', label='Medido no ponto 9')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.title(f"Pressão no pixel poi9")
plt.legend
plt.show()

'''
plt.figure(3)
plt.plot(t, v_at_transducer, 'r', label='Medido na fonte')
plt.plot(t, v_at_poi1, 'g', label='Medido em outro pixel antes da interface')
plt.title(f"Pressão no pixel do transdutor vs no ponto de interesse - Lap {deriv_accuracy}")
plt.legend()
plt.show()

plt.figure(4)
plt.plot(t, v_at_poi2, 'r', label='Medido no segundo meio')
plt.plot(t, v_at_poi3, 'g', label='Medido em outro pixel no segundo meio')
plt.title(f"Pressão nos pontos de interesse - Lap {deriv_accuracy}")
plt.legend()
plt.show()
'''

'''
print(f"Second Media {soundspeed2}")
print(f"Lap 2 - Measured Wave Speed: {u_wave_speed2pi} pixels/timestep")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {v_wave_speed2pi} pixels/timestep")
print(f"Lap 2 - Measured Wave Speed: {u_wave_speed2} m/s")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {v_wave_speed2} m/s\n\n")
'''