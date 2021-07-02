import numpy as np

def circular_wave(x, freq, mag):
    dist = np.linalg.norm(x)
    return np.sin(dist*freq)*mag*x/dist

def neg(func):
    return lambda x : -func(x)

def concentric(x):
    freq = np.pi/2
    mag = 2.5
    return +2*np.sin(x*np.pi*1.5) - circular_wave(x, freq, mag) - x/5
