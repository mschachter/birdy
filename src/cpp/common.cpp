#include "common.h"

void run_ode(double** output, double* initial_state, double duration, double dt, gsl_odeiv_system* sys)
{
    double t = 0.0;
    double t1 = dt;
    double start_step = 1e-10;
    double next_state[2] = {initial_state[0], initial_state[1]};

    const gsl_odeiv_step_type* T = gsl_odeiv_step_rk8pd;
	gsl_odeiv_step* s = gsl_odeiv_step_alloc(T, 2);
	gsl_odeiv_control* c = gsl_odeiv_control_y_new(1e-10, 0.0);
	gsl_odeiv_evolve* e = gsl_odeiv_evolve_alloc(2);

	int step = 0;

    while (t < duration)
    {
        while (t < t1)
        {
            int status = gsl_odeiv_evolve_apply(e, c, s, sys, &t, t1, &start_step, next_state);
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
