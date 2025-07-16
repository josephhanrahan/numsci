#include <stdlib.h>
#include <math.h>

/* Levenberg-Marquardt solver */
void LMDER(int(*fcn)(int* m, int* n,double* x, double* fvec, double* fjac,
           int* ldfjac, int* iflag),
           const int m, const int n, double* x, double* fvec, double* fjac,
           const int ldfjac, const double ftol, const double xtol, const double gtol,
           const int maxfev, double* diag, const int mode, const double factor,
           const int nprint, int* info, int* nfev, int* njev, int* ipvt,
           double* qtf, double* wa1, double* wa2, double* wa3, double* wa4);

void LMDIF(int(*fcn)(int* m, int* n, double* x, double* fvec, int* iflag),
           const int m, const int n, double* x, double* fvec, const double ftol,
           const double xtol, const double gtol, const int maxfev, const double epsfcn,
           double* diag, const int mode, const double factor, const int nprint,
           int* info, int* nfev, double* fjac, const int ldfjac, int* ipvt,
           double* qtf, double* wa1, double* wa2, double* wa3, double* wa4);