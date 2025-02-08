import numpy as np
from pid import PID


class GoPiGo3:
    def __init__(
        self,
        # Simulation Object
        simulation_object=None,
        # CoppeliaSim Handle
        handle_name="/GoPiGo3",
        target_handle_name="/Target",
        # Controllers
        genome=np.zeros(6),
    ):
        # Simulation Object
        self.sim = simulation_object

        # Handles
        self.handle = self.sim.getObject(handle_name)
        self.target_handle = self.sim.getObject(target_handle_name)
        self.motor_L_handle = self.sim.getObject(handle_name + "/LeftJoint")
        self.motor_R_handle = self.sim.getObject(handle_name + "/RightJoint")

        # Controllers
        self.OrientationController = PID(
            K_p=genome[0],
            K_i=genome[1],
            K_d=genome[2],
            dt=self.sim.getSimulationTimeStep(),
        )
        self.VelocityController = PID(
            K_p=genome[3],
            K_i=genome[4],
            K_d=genome[5],
            dt=self.sim.getSimulationTimeStep(),
        )

        # Differential Drive Model Parameters
        self.r = 0.0325
        self.s = 0.115

        # Metrics
        self.not_converged = True
        self.convergence_time = np.inf
        self.convergence_distance = np.inf
        self.target_distance = np.inf
        self.max_roll = 0.0
        self.max_pitch = 0.0

        # Genetic Algorithm Parameters
        self.genome = genome
        self.fitness = -np.inf

    def sense_and_actuate(self):
        timestamp = self.sim.getSimulationTime()

        # Getting GoPiGo3 Differential Drive Model Position
        position = np.mean(
            [
                self.sim.getObjectPosition(self.motor_L_handle),
                self.sim.getObjectPosition(self.motor_R_handle),
            ],
            axis=0,
        )[:2]

        # Vectors and distances
        front_vector = np.array(self.sim.getObjectMatrix(self.handle)).reshape(3, 4)[
            :2, 0
        ]
        roll, pitch, _ = np.abs(self.sim.getObjectOrientation(self.handle))
        path_vector = (
            np.array(self.sim.getObjectPosition(self.target_handle))[:2] - position
        )
        target_distance = np.linalg.norm(path_vector)
        target_vector = path_vector / target_distance

        # Error signals
        orientation_error = np.cross(front_vector, target_vector)
        position_error = np.dot(front_vector, target_vector) * target_distance

        # Controllers
        theta_dot = self.OrientationController.get_output(orientation_error)
        v_B = self.VelocityController.get_output(position_error)

        # Differential Drive Model
        theta_dot_L, theta_dot_R = np.linalg.inv(
            [[self.r / 2.0, self.r / 2.0], [-self.r / self.s, self.r / self.s]]
        ) @ np.array([v_B, theta_dot])

        # Update target distance
        self.target_distance = target_distance

        # Update max roll and max pitch
        if self.max_roll < roll:
            self.max_roll = roll
        if self.max_pitch < pitch:
            self.max_pitch = pitch

        # Convergence criteria
        self.not_converged = self.target_distance > 0.01

        # Stop condition
        if self.not_converged:
            # Move joints
            self.sim.setJointTargetVelocity(self.motor_L_handle, theta_dot_L)
            self.sim.setJointTargetVelocity(self.motor_R_handle, theta_dot_R)

            # Update convergence time
            self.convergence_time = timestamp

        else:
            # Move joints
            self.sim.setJointTargetVelocity(self.motor_L_handle, 0.0)
            self.sim.setJointTargetVelocity(self.motor_R_handle, 0.0)

    def get_metrics(self):
        # Returning vectorized performance parameters
        return np.array(
            [
                self.not_converged,
                self.convergence_time,
                self.target_distance,
                self.max_roll,
                self.max_pitch,
            ]
        )
