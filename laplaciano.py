# Laplacianos com diferentes acurácias
# Felipe Derewlany Gutierrez

import numpy as np


def fdm_laplaciano(matrix, precision=2):
    M = matrix.shape[0]
    N = matrix.shape[1]

    if(precision == 2):
        lap = np.zeros((M+2, N+2))
        lap[1:-1, 1:-1] = -4 * matrix

        lap[0:-2, 1:-1] += matrix
        lap[2:, 1:-1] += matrix

        lap[1:-1, 2:] += matrix
        lap[1:-1, 0:-2] += matrix

        y = lap[1:-1, 1:-1]
        return y
    elif(precision == 4):
        #f_xx = (-1*f[i-2]+16*f[i-1]-30*f[i+0]+16*f[i+1]-1*f[i+2])/(12*1.0*h**2)
        M = matrix.shape[0]
        N = matrix.shape[1]
        lap = np.zeros((M+4, N+4))
        lap[2:-2, 2:-2] = -60 * matrix

        lap[0:-4, 2:-2] += -1 * matrix
        lap[1:-3, 2:-2] += 16 * matrix
        lap[3:-1, 2:-2] += 16 * matrix
        lap[4:, 2:-2] += -1 * matrix

        lap[2:-2, 0:-4] += -1 * matrix
        lap[2:-2, 1:-3] += 16 * matrix
        lap[2:-2, 3:-1] += 16 * matrix
        lap[2:-2, 4:] += -1 * matrix

        y = lap[2:-2, 2:-2]
        return y/12
    elif(precision == 6):
        #f_xx = (2*f[i-3]-27*f[i-2]+270*f[i-1]-490*f[i+0]+270*f[i+1]-27*f[i+2]+2*f[i+3]) \
        # / (180*1.0*h**2)
        M = matrix.shape[0]
        N = matrix.shape[1]
        lap = np.zeros((M+6, N+6))
        lap[3:-3, 3:-3] = -980 * matrix

        lap[0:-6, 3:-3] += 2 * matrix
        lap[1:-5, 3:-3] += -27 * matrix
        lap[2:-4, 3:-3] += 270 * matrix
        lap[4:-2, 3:-3] += 270 * matrix
        lap[5:-1, 3:-3] += -27 * matrix
        lap[6:, 3:-3] += 2 * matrix

        lap[3:-3, 0:-6] += 2 * matrix
        lap[3:-3, 1:-5] += -27 * matrix
        lap[3:-3, 2:-4] += 270 * matrix
        lap[3:-3, 4:-2] += 270 * matrix
        lap[3:-3, 5:-1] += -27 * matrix
        lap[3:-3, 6:] += 2 * matrix

        y = lap[3:-3, 3:-3]
        return y/180

    elif (precision == 8):
        # f_xx = (-9*f[i-4]+128*f[i-3]-1008*f[i-2]+8064*f[i-1]-14350*f[i+0] \
        # +8064*f[i+1]-1008*f[i+2]+128*f[i+3]-9*f[i+4])/(5040*1.0*h**2)
        M = matrix.shape[0]
        N = matrix.shape[1]
        lap = np.zeros((M + 8, N + 8))
        lap[4:-4, 4:-4] = -28700 * matrix

        lap[0:-8, 4:-4] += -9 * matrix
        lap[1:-7, 4:-4] += 128 * matrix
        lap[2:-6, 4:-4] += -1008 * matrix
        lap[3:-5, 4:-4] += 8064 * matrix
        lap[5:-3, 4:-4] += 8064 * matrix
        lap[6:-2, 4:-4] += -1008 * matrix
        lap[7:-1, 4:-4] += 128 * matrix
        lap[8:, 4:-4] += -9 * matrix

        lap[4:-4, 0:-8] += -9 * matrix
        lap[4:-4, 1:-7] += 128 * matrix
        lap[4:-4, 2:-6] += -1008 * matrix
        lap[4:-4, 3:-5] += 8064 * matrix
        lap[4:-4, 5:-3] += 8064 * matrix
        lap[4:-4, 6:-2] += -1008 * matrix
        lap[4:-4, 7:-1] += 128 * matrix
        lap[4:-4, 8:] += -9 * matrix

        y = lap[4:-4, 4:-4]
        return y/5040

    elif (precision == 10):
        # f_xx = (8*f[i-5]-125*f[i-4]+1000*f[i-3]-6000*f[i-2]+42000*f[i-1]-73766*f[i+0] \
        # +42000*f[i+1]-6000*f[i+2]+1000*f[i+3]-125*f[i+4]+8*f[i+5])/(25200*1.0*h**2)
        M = matrix.shape[0]
        N = matrix.shape[1]
        lap = np.zeros((M + 10, N + 10))
        lap[5:-5, 5:-5] = -147532 * matrix

        lap[0:-10, 5:-5] += 8 * matrix
        lap[1:-9, 5:-5] += -125 * matrix
        lap[2:-8, 5:-5] += 1000 * matrix
        lap[3:-7, 5:-5] += -6000 * matrix
        lap[4:-6, 5:-5] += 42000 * matrix
        lap[6:-4, 5:-5] += 42000 * matrix
        lap[7:-3, 5:-5] += -6000 * matrix
        lap[8:-2, 5:-5] += 1000 * matrix
        lap[9:-1, 5:-5] += -125 * matrix
        lap[10:, 5:-5] += 8 * matrix

        lap[5:-5, 0:-10] += 8 * matrix
        lap[5:-5, 1:-9] += -125 * matrix
        lap[5:-5, 2:-8] += 1000 * matrix
        lap[5:-5, 3:-7] += -6000 * matrix
        lap[5:-5, 4:-6] += 42000 * matrix
        lap[5:-5, 6:-4] += 42000 * matrix
        lap[5:-5, 7:-3] += -6000 * matrix
        lap[5:-5, 8:-2] += 1000 * matrix
        lap[5:-5, 9:-1] += -125 * matrix
        lap[5:-5, 10:] += 8 * matrix

        y = lap[5:-5, 5:-5]
        return y/25200

    elif (precision == 12):
        # Acurácia 12
        M = matrix.shape[0]
        N = matrix.shape[1]
        lap = np.zeros((M + 12, N + 12))

        lap[6:-6, 6:-6] = -5.84788263 * matrix

        lap[0:-12, 6:-6] += 3.55076291e-6 * matrix
        lap[1:-11, 6:-6] += 2.74851163e-4 * matrix
        lap[2:-10, 6:-6] += -4.72596711e-3 * matrix
        lap[3:-9, 6:-6] += 3.89013718e-2 * matrix
        lap[4:-8, 6:-6] += -2.36337610e-1 * matrix
        lap[5:-7, 6:-6] += 1.66385446 * matrix
        lap[7:-5, 6:-6] += 1.66385446 * matrix
        lap[8:-4, 6:-6] += -2.36337610e-1 * matrix
        lap[9:-3, 6:-6] += 3.89013718e-2 * matrix
        lap[10:-2, 6:-6] += -4.72596711e-3 * matrix
        lap[11:-1, 6:-6] += 2.74851163e-4 * matrix
        lap[12:, 6:-6] += 3.55076291e-6 * matrix

        lap[6:-6, 0:-12] += 3.55076291e-6 * matrix
        lap[6:-6, 1:-11] += 2.74851163e-4 * matrix
        lap[6:-6, 2:-10] += -4.72596711e-3 * matrix
        lap[6:-6, 3:-9] += 3.89013718e-2 * matrix
        lap[6:-6, 4:-8] += -2.36337610e-1 * matrix
        lap[6:-6, 5:-7] += 1.66385446 * matrix
        lap[6:-6, 7:-5] += 1.66385446 * matrix
        lap[6:-6, 8:-4] += -2.36337610e-1 * matrix
        lap[6:-6, 9:-3] += 3.89013718e-2 * matrix
        lap[6:-6, 10:-2] += -4.72596711e-3 * matrix
        lap[6:-6, 11:-1] += 2.74851163e-4 * matrix
        lap[6:-6, 12:] += 3.55076291e-6 * matrix

        y = lap[6:-6, 6:-6]
        return y

    else:
        print("Invalid Precision")
        exit(2)
