import numpy as np
import matplotlib.pyplot as plt
from ..config import config
from .draw_text import *

def im2matrix(im):
    spotlight_func = spotlight_rectangle
    step_func = step_char_wise

    xsteps, step_size = step_func(im)
    _, him = im.size
    nsteps = len(xsteps)

    Mr = him * step_size
    Mc = nsteps
    X = np.zeros(shape=(Mr, Mc))
    for col_i, xo in enumerate(xsteps):
        # Note np_im is a matrix, it follows the matrix (row, column) convention
        fim = spotlight_func(np.array(im), (0, xo, him, step_size))
        fim_vec = np.reshape(fim, [-1,1]).flatten()
        # plt.plot(fim_vec)
        # plt.show()
        X[:,col_i] = fim_vec
    return X, (im.size[0], step_size)

def text2matrix(text):
    im = text2bitmap(text)    
    X, (h, w) = im2matrix(im)
    return X, (h, w)




