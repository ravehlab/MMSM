import numpy as np

def circular_wave(x, freq, mag):
    dist = np.linalg.norm(x)
    return np.sin(dist*freq)*mag*x/dist

def neg(func):
    return lambda x : -func(x)

def concentric(x):
    freq = np.pi/2
    mag = 3.#2.5
    return +2*np.sin(x*np.pi*1.5) - circular_wave(x, freq, mag) - x/5

def get_concentric_with_params(square_freq=np.pi*1.5, square_mag=2, circular_freq=np.pi/2, circular_mag=3):
    def concentric_with_params(x):
        return (
                square_mag*np.sin(x*square_freq) -              # eggshels
                circular_wave(x, circular_freq, circular_mag) - # concentric waves
                x/5                                             # bowl
                )
    return concentric_with_params

