# numsci

## num[ba]sci[py] - A Numba-Compatible Port for Popular Scipy Functions

Most Scipy Functions do not work in jit-compiled Numba functions. This project aims to make this possible.

### How to use

Model functions must abide by the following convention

```
from numsci.optimize import model_sig

@model_sig
def model_function(x: float64, params: ndarray(dtype=float64)) -> float64:
    ***do computation***
    return result
```

Where the @model_sig decorator indicates that the function is a Numba Cfunc of signature `float64(float64, *float64)`.

Outside of the jit-compiled function, the caller must also obtain the address to the model function

```
model_function_address = model_function.address
```

This can then be used to call Scipy-like functions like `curve_fit()` in a `@njit` decorated function


```
from numsci.optimize import curve_fit

@njit
def njit_function():
    fvec, pcov = curve_fit(model_function_address, xdata, ydata)
```

### Setup

Install dependencies and compile

```
pip install -r requirements.txt
bash build.sh                                          
```