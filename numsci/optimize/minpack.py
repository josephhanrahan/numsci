from numba import cfunc, types, njit
import os
import ctypes
import numpy as np

minpack = ctypes.CDLL(os.getcwd() + "/minpack.so")

_lmdif = minpack.LMDIF
_lmdif.restype = None
_lmdif.argtypes = _lmdif.argtypes = [ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                   ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, 
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

@njit(cache=True)
def lmdif():
    pass

_set_variables = minpack.set_variables
_set_variables.restype = None
_set_variables.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

@njit(cache=True)
def set_variables(f_addr, xdata, ydata):
    _set_variables(f_addr, xdata.ctypes.data, ydata.ctypes.data)
    return

_residual_function = minpack.residual_function
_residual_function_restype = ctypes.c_int
_residual_function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
residual_function_addr = ctypes.cast(_residual_function, ctypes.c_void_p).value
    
@njit(cache=True)
def leastsq(func_addr, x0, m, Dfun=None, full_output=False,
            col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    pass
