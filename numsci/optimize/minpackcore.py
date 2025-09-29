from numba import types, njit
import sysconfig
import ctypes
import numpy as np

minpack = ctypes.CDLL(sysconfig.get_paths()['platlib'] 
                      + "/numsci/optimize/libminpack.so")

model_sig = types.double(types.double, types.CPointer(types.double))

_lmdif = minpack.LMDIF
_lmdif.restype = None
_lmdif.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, 
                   ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                   ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int, 
                   ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, 
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, 
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_set_variables = minpack.set_variables
_set_variables.restype = None
_set_variables.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

@njit
def set_variables(f_addr, xdata, ydata):
    """
    A wrapper for C callback function _set_variables(), which sets the the global
    variables global_function_ptr, global_xdata, global_ydata in residual.c to be
    used when calling residual_function() in residual.c, thus bypassing the need
    to dynamically create a new residual function for minpack calls.

    Parameters
    ----------
    f_addr : address to callable cfunc
        The address to model function, f(x, *params). It must take the independent
        variable x as the first argument and an array of params as the second
        argument.
    xdata : np.ndarray
        The independent variable where the data is measured in the form of a
        length M array.
        
    ydata : np.ndarray
        The dependent data, a length M array - nominally ``f(xdata, *params)``.

    Returns
    -------
    residual_function_addr : ctypes.c_void_p
        A pointer to the address of the residual function in residual.c
    """
    _set_variables(f_addr, xdata.ctypes.data, ydata.ctypes.data)
    return residual_function_addr

_residual_function = minpack.residual_function
_residual_function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                               ctypes.c_void_p, ctypes.c_void_p]
residual_function_addr = ctypes.cast(_residual_function, ctypes.c_void_p).value
    
@njit
def leastsq(func_addr, x0, m, Dfun=None, full_output=False,
            col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    x0_len = len(x0)
    n = types.intc(x0_len)
    m = types.intc(m)
    # initial params (x0), length n
    x = np.array(x0, dtype=np.float64)
    # output array of residuals, length m
    fvec = np.zeros((m,), dtype=np.float64)

    # set the rest of the parameters lmdif wants
    ftol=types.float64(1.49012e-8)
    xtol=types.float64(1.49012e-8)
    gtol=types.float64(0.0)
    if maxfev == 0:
        maxfev = 200*(n+1)
    maxfev=types.intc(maxfev)
    epsfcn=types.float64(0.0)
    diag = np.ones((n,), dtype=np.float64)
    mode=types.intc(0)
    factor=types.float64(1.0e2)
    nprint=types.intc(0)
    info=np.zeros(1, dtype=np.intc)
    nfev=np.array(0, dtype=np.intc)
    fjac=np.zeros((n, m), dtype=np.float64)
    ldfjac=types.intc(m)
    ipvt = np.zeros((n,), dtype=np.intc)
    qtf=np.zeros((n,), dtype=np.float64)
    wa1=np.zeros((n,), dtype=np.float64)
    wa2=np.zeros((n,), dtype=np.float64)
    wa3=np.zeros((n,), dtype=np.float64)
    wa4=np.zeros((m,), dtype=np.float64)

    if Dfun is None:
        _lmdif(func_addr, m, n, x.ctypes.data, fvec.ctypes.data, ftol, xtol, gtol, 
               maxfev, epsfcn, diag.ctypes.data, mode, factor, nprint, 
               info.ctypes.data, nfev.ctypes.data, fjac.ctypes.data, ldfjac, 
               ipvt.ctypes.data, qtf.ctypes.data, wa1.ctypes.data, wa2.ctypes.data, 
               wa3.ctypes.data, wa4.ctypes.data)

    return x, fvec

@njit
def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=None,
              check_finite=None, bounds=(-np.inf, np.inf), method=None,
              jac=None, full_output=False, nan_policy=None):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``.

    Parameters
    ----------
    f : address to callable cfunc
        The address to model function, f(x, *params). It must take the independent
        variable x as the first argument and an array of params to fit
        as the second argument.
    xdata : ndarray, float64
        The independent variable where the data is measured in the form of a
        length M array.
    ydata : ndarray, float64
        The dependent data, a length M array - nominally ``f(xdata, *params)``.
    p0 : ndarray, float64
        Initial guess for the parameters (length N).

    Returns
    -------
    popt : ndarray, float64
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized.
    """

    set_variables(f, xdata, ydata)

    method = 'lm'

    if method == 'lm':
        popt, fvec = leastsq(residual_function_addr, p0, len(xdata))

    return popt, None