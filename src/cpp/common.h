#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

void run_ode(double** output, double* initial_state, double duration, double dt, gsl_odeiv2_system* sys);

#endif

