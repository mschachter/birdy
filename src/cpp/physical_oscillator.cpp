#include "common.h"
#include "physical_oscillator.h"

PhysicalParams* physical_oscillator_init(double k1, double psub, double f0)
{
    PhysicalParams* pp = new PhysicalParams();
    pp->m = 4e-10;
    pp->k1 = k1;
    pp->k2 = 400.0;
    pp->beta1 = 444e-7;
    pp->beta2 = 4e-11;
    pp->c = 16e-3;
    pp->f0 = f0;
    pp->alab = 2e-4;
    pp->tau = 5e-6;
    pp->a01 = 0.1;
    pp->a02 = 0.11;
    pp->psub = psub;

    return pp;
}

int physical_oscillator_rhs(double t, const double state[], double dstate[], void* params)
{
    PhysicalParams* pp = (PhysicalParams*) params;

    double x = state[0];
    double x2 = x*x;
    double x3 = x*x*x;
    double v = state[1];
    double v3 = v*v*v;

    double da = pp->a01 - pp->a02;

    double tterm = (da + 2*pp->tau*v) / (pp->a01 + x + pp->tau*v);
    dstate[0] = v;
    dstate[1] = -pp->k1*x - pp->k2*x3 - pp->beta1*v - pp->beta2*v3 - pp->c*x2*v + pp->f0 + pp->alab*pp->psub*tterm;
    dstate[1] /= pp->m;

    return GSL_SUCCESS;
}

int physical_oscillator_jacobian(double t, const double state[], double* d2state, double* d2t, void* params)
{
    gsl_matrix_view d2state_mat = gsl_matrix_view_array(d2state, 2, 2);
	gsl_matrix* m = &d2state_mat.matrix;
	gsl_matrix_set(m, 0, 0, 0.0);
	gsl_matrix_set(m, 0, 1, 0.0);
	gsl_matrix_set(m, 1, 0, 0.0);
	gsl_matrix_set(m, 1, 1, 0.0);
	d2t[0] = 0.0;
	d2t[1] = 0.0;

	return GSL_SUCCESS;
}

void physical_oscillator_run(double** output, double* initial_state, double duration, double dt, PhysicalParams* pp)
{
	gsl_odeiv2_system sys = {physical_oscillator_rhs, physical_oscillator_jacobian, 2, pp};
    run_ode(output, initial_state, duration, dt, &sys);
}
