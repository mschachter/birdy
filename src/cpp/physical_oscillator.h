#ifndef PHYSICAL_OSCILLATOR_H
#define PHYSICAL_OSCILLATOR_H

typedef struct
{
    double m;
    double k1;
    double k2;
    double beta1;
    double beta2;
    double c;
    double f0;
    double alab;
    double tau;
    double a01;
    double a02;
    double psub;

} PhysicalParams;

PhysicalParams* physical_oscillator_init(double k1, double psub, double f0);
int physical_oscillator_rhs(double t, const double state[], double dstate[], void* params);
int physical_oscillator_jacobian(double t, const double state[], double* d2state, double* d2t, void* params);
void physical_oscillator_run(double** output, double* initial_state, double duration, double dt, PhysicalParams* pp);

#endif
