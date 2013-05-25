#ifndef NORMAL_OSCILLATOR_H
#define NORMAL_OSCILLATOR_H

typedef struct
{
    double alpha;
    double beta;
    double gamma;

} NormalParams;

NormalParams* normal_oscillator_init(double alpha, double beta);
void normal_oscillator_delete(NormalParams* pp);
int normal_oscillator_rhs(double t, const double state[], double dstate[], void* params);
int normal_oscillator_jacobian(double t, const double state[], double* d2state, double* d2t, void* params);
void normal_oscillator_run(double** output, double* initial_state, double duration, double dt, NormalParams* pp);
double normal_oscillator_nullcline_x(double* state, NormalParams* pp);
double normal_oscillator_nullcline_dx(double* state, NormalParams* pp);

#endif
