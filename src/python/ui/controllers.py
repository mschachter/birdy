from ui.models import SimulationOutput


class MainController(object):

    def __init__(self):
        pass

    def run_simulation(self, oscillator_model, duration, dt, initial_x=0.0, initial_v=0.0):
        params = dict()
        for pname,cparam in oscillator_model.get_control_params().iteritems():
            params[pname] = cparam.current_val

        output = oscillator_model.oscillator.run_simulation(params, duration, dt, initial_x=initial_x, initial_v=initial_v)
        return SimulationOutput(output, duration, dt)
