

typedef double (*model_function)(double, double*);

void* global_function_ptr;
double* global_xdata;
double* global_ydata;

void set_variables(void* function_ptr, double* xdata, double* ydata) {
    global_function_ptr = function_ptr;
    global_xdata = xdata;
    global_ydata = ydata;
    return;
}

int residual_function(int* m, int* n, double* x, double* fvec, int* iflag) {
    model_function f = (model_function)global_function_ptr;

    for (int i=0; i<*m; i++) {
        fvec[i] = f(global_xdata[i], x) - global_ydata[i];
    }
    return 0;
}

