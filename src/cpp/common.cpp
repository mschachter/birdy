#include "common.h"

/*
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
            printf("t=%0.9f, step=%0.9f\n", t, )
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
*/

void run_ode(double** output, double* initial_state, double duration, double dt, gsl_odeiv2_system* sys)
{
    double t = 0.0;
    double t1 = dt;
    double start_step = dt;
    double current_step = start_step;
    double next_state[2] = {initial_state[0], initial_state[1]};
    int max_steps = (int) ceil(duration / dt);

    const gsl_odeiv2_step_type* T = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step* s = gsl_odeiv2_step_alloc(T, 2);

	gsl_odeiv2_control* c = gsl_odeiv2_control_y_new(1e-8, 1e-10);
	gsl_odeiv2_evolve* e = gsl_odeiv2_evolve_alloc(2);

	int step = 0;
	int substeps = 0;
	double mean_step_size = 0.0;

    while (t < duration && step < max_steps)
    {
        substeps = 0;
        mean_step_size = 0.0;
        while (t < t1)
        {
            current_step = start_step;
            int status = gsl_odeiv2_evolve_apply(e, c, s, sys, &t, t1, &current_step, next_state);
            mean_step_size += current_step;
            substeps++;
            if (status != GSL_SUCCESS) {
                printf("FAIL Step %d: t=%e, t1=%e dt=%e, current_step=%e\n", step, t, t1, dt, current_step);
                break;
            }
        }
        mean_step_size /= substeps;
        t1 += dt;
        //printf("step %d: t=%e, t1=%e, start_step=%e, mean_step=%e, substeps=%d\n", step, t, t1, start_step, mean_step_size, substeps);
        output[step][0] = next_state[0];
        output[step][1] = next_state[1];
        step++;
    }

    gsl_odeiv2_evolve_free(e);
	gsl_odeiv2_control_free(c);
	gsl_odeiv2_step_free(s);
}
