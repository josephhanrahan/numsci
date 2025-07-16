

extern void* global_function_ptr;
extern double* global_xdata;
extern double* global_ydata;

void set_variables(void* function_ptr, double* xdata, double* ydata);

int residual_function(int* m, int* n, double* x, double* fvec, int* iflag);