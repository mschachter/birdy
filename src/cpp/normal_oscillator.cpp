#include "common.h"
#include "normal_oscillator.h"

NormalParams* normal_oscillator_init(double alpha, double beta)
{
    NormalParams* pp = new NormalParams();
    pp->gamma = 23500.0;
    pp->alpha = alpha;
    pp->beta = beta;

    return pp;
}

void normal_oscillator_delete(NormalParams* pp)
{
    delete pp;
}

int normal_oscillator_rhs(double t, const double state[], double dstate[], void* params)
{
    NormalParams* pp = (NormalParams*) params;

    double x = state[0];
    double x2 = x*x;
    double x3 = x*x*x;
    double v = state[1];

    double gamma = pp->gamma;
    double gamma2 = gamma*gamma;

    dstate[0] = v;
    dstate[1] = gamma2*pp->alpha + gamma2*pp->beta*x + gamma2*x2 - gamma*x*v - gamma2*x3 - gamma*x2*v;

    return GSL_SUCCESS;
}

int normal_oscillator_jacobian(double t, const double state[], double* d2state, double* d2t, void* params)
{
    NormalParams* pp = (NormalParams*) params;

    double x = state[0];
    double x2 = x*x;
    double v = state[1];

    double gamma = pp->gamma;
    double gamma2 = gamma*gamma;

    double J21 = gamma2*pp->beta + 2*gamma2*x - gamma*v - 3*gamma*x2 - 2*gamma*x*v;
    double J22 = -gamma*x - gamma*x2;

    gsl_matrix_view d2state_mat = gsl_matrix_view_array(d2state, 2, 2);
	gsl_matrix* m = &d2state_mat.matrix;
	gsl_matrix_set(m, 0, 0, 0.0);
	gsl_matrix_set(m, 0, 1, 1.0);
	gsl_matrix_set(m, 1, 0, J21);
	gsl_matrix_set(m, 1, 1, J22);
	d2t[0] = 0.0;
	d2t[1] = 0.0;

	return GSL_SUCCESS;
}

void normal_oscillator_run(double** output, double* initial_state, double duration, double dt, NormalParams* pp)
{
    gsl_odeiv2_system sys = {normal_oscillator_rhs, normal_oscillator_jacobian, 2, pp};
    run_ode(output, initial_state, duration, dt, &sys);
}

double normal_oscillator_nullcline_x(double* state, NormalParams* pp)
{
    double x = state[0];
    double x2 = x*x;
    double x3 = x*x*x;

    double gamma = pp->gamma;
    double gamma2 = gamma*gamma;

    return gamma2*pp->alpha + gamma2*pp->beta*x + gamma2*x2 - gamma*x3;
}

double normal_oscillator_nullcline_dx(double* state, NormalParams* pp)
{
    double x = state[0];
    double x2 = x*x;

    double gamma = pp->gamma;
    double gamma2 = gamma*gamma;

    return gamma2*pp->beta + 2*gamma2*x - 3*gamma*x2;
}