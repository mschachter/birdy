#include <stdio.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>

#include "physical_oscillator.h"

PhysicalParams* physical_oscillator_init(double k1, double psub, double f0)
{
    PhysicalParams* pp = new PhysicalParams();
    pp->m = 4e-10;
    pp->k1 = k1;
    pp->k2 = 400;
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

void test_function(double** x, int size)
{
    printf("in test function\n");
    for (int k = 0; k < size; k++) {
        printf("x[%d, 0]=%0.6f\n", k, x[k][0]);
        printf("x[%d, 1]=%0.6f\n", k, x[k][1]);
        x[k][1] = x[k][1]*x[k][0];
    }
}

void physical_oscillator_run(double** output, double* initial_state, double duration, double dt, PhysicalParams* pp)
{
    double t = 0.0;
    double t1 = dt;
    double start_step = 1e-10;
    double next_state[2] = {initial_state[0], initial_state[1]};
    printf("dt=%0.6f, initial_x=%0.6f, initial_v=%0.6f, k1=%0.6f, psub=%0.6f, f0=%0.6f\n",
           dt, initial_state[0], initial_state[1], pp->k1, pp->psub, pp->f0);

    const gsl_odeiv_step_type* T = gsl_odeiv_step_rk8pd;
	gsl_odeiv_step* s = gsl_odeiv_step_alloc(T, 2);
	gsl_odeiv_control* c = gsl_odeiv_control_y_new(1e-10, 0.0);
	gsl_odeiv_evolve* e = gsl_odeiv_evolve_alloc(2);
	gsl_odeiv_system sys = {physical_oscillator_rhs, physical_oscillator_jacobian, 2, pp};

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

/*
int main(int nargs, char** args)
{
    double duration = 0.050;
    double dt = 1e-6;
    double k1 = 0.016;
    double psub = 1900.0;
    double f0 = 0.0399;

    int nsteps = ceil(duration / dt);
    double** output = new double*[nsteps];
    for (int k = 0; k < nsteps; k++) {
        output[k] = new double[2];
    }

    double* initial_state = new double[2];
    initial_state[0] = 0.0;
    initial_state[1] = 0.0;

    PhysicalParams* pp = physical_oscillator_init(k1, psub, f0);

    physical_oscillator_run(output, initial_state, duration, dt, pp);

    for (int k = 0; k < nsteps; k++) {
        printf("Step %d: x=%0.6f, v=%0.6f\n", k, output[k][0], output[k][1]);
    }

    return 0;
}
*/