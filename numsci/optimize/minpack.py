from numba import cfunc, types, njit
import os
import ctypes
import numpy as np

minpack = ctypes.CDLL(os.getcwd() + "/minpack.so")
