#include <stdio.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>

#include "normal_oscillator.h"

NormalParams* normal_oscillator_init(double alpha, double beta)
{
    NormalParams* pp = new NormalParams();
    pp->gamma = 23500.0;
    pp->alpha = alpha;
    pp->beta = beta;

    return pp;
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
    dstate[1] = gamma2*pp->alpha + gamma2*pp->beta*x + gamma2*x2 - gamma*x*v - gamma*x3 - gamma*x2*v;

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
    double t = 0.0;
    double t1 = dt;
    double start_step = 1e-10;
    double next_state[2] = {initial_state[0], initial_state[1]};
    printf("dt=%0.6f, initial_x=%0.6f, initial_v=%0.6f, gamma=%0.6f, alpha=%0.6f, beta=%0.6f\n",
           dt, initial_state[0], initial_state[1], pp->gamma, pp->alpha, pp->beta);

    const gsl_odeiv_step_type* T = gsl_odeiv_step_rk8pd;
	gsl_odeiv_step* s = gsl_odeiv_step_alloc(T, 2);
	gsl_odeiv_control* c = gsl_odeiv_control_y_new(1e-10, 0.0);
	gsl_odeiv_evolve* e = gsl_odeiv_evolve_alloc(2);
	gsl_odeiv_system sys = {normal_oscillator_rhs, normal_oscillator_jacobian, 2, pp};

	int step = 0;

    while (t < duration)
    {
        while (t < t1)
        {
            int status = gsl_odeiv_evolve_apply(e, c, s, &sys, &t, t1, &start_step, next_state);
            if (status != GSL_SUCCESS)
                break;
        }
        t1 += dt;
        output[step][0] = next_state[0];
        output[step][1] = next_state[1];
        step++;
    }

    gsl_odeiv_evolve_free(e);
	gsl_odeiv_control_free(c);
	gsl_odeiv_step_free(s);
}
