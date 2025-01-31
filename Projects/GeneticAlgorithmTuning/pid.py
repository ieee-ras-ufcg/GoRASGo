# PID Controller
class PID:
    def __init__(
        self,
        # PID Parameters
        K_p=0.0,  # Proportional parameter
        K_i=0.0,  # Integral parameter
        K_d=0.0,  # Differential parameter
        dt=50e-3,  # Differential time step
    ):
        # Simulation Time Step
        self.dt = dt

        # PID Gains
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        # Error signals
        self.prev_e = None
        self.curr_e = 0.0
        self.accu_e = 0.0
        self.diff_e = 0.0

    def get_output(self, e):
        # Update error signals
        if self.prev_e is None:
            self.prev_e = e

        self.curr_e = e
        self.accu_e += e * self.dt
        self.diff_e = (self.curr_e - self.prev_e) / self.dt

        # Compute output
        output = self.K_p * e + self.K_i * self.accu_e + self.K_d * self.diff_e

        # Update previous error
        self.prev_e = self.curr_e

        return output

    def reset(self):
        self.prev_e = None
        self.curr_e = 0.0
        self.accu_e = 0.0
        self.diff_e = 0.0