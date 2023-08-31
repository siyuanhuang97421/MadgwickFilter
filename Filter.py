import numpy as np

def quat_product(q1,q2):
    q1_x = q1[0]
    q1_y = q1[1]
    q1_z = q1[2]
    q1_w = q1[3]

    q2_x = q2[0]
    q2_y = q2[1]
    q2_z = q2[2]
    q2_w = q2[3]

    return np.array([q1_x*q2_x - q1_y*q2_y - q1_z*q2_z - q1_w*q2_w,
                     q1_x*q2_y + q1_y*q2_x + q1_z*q2_w - q1_w*q2_z,
                     q1_x*q2_z - q1_y*q2_w + q1_z*q2_x + q1_w*q2_y,
                     q1_x*q2_w + q1_y*q2_z - q1_z*q2_y + q1_w*q2_x])


class Filter():
    def __init__(self, initial_states):
        self.current_state = initial_states
        self.t = 0.0
        self.current_measurement = None

    def updateIMUMeasurement(self, imu_measurement):
        self.imu_measurement = imu_measurement

    def step(self):
        raise NotImplementedError
    
    def getCurrentState(self):
        return self.current_state
    
class MadgwickFilter(Filter):
    def step(self):
        beta = 0.1
        gyro_measurement = np.append(np.array([0]),self.imu_measurement["omega_xyz"])
        accel_measurement = np.append(np.array([0]),self.imu_measurement["accel_xyz"])
        accel_measurement /= np.linalg.norm(accel_measurement)

        q_xyzw = self.current_state["q_xyzw"]
        q_x = q_xyzw[0]
        q_y = q_xyzw[1]
        q_z = q_xyzw[2]
        q_w = q_xyzw[3]

        f = np.array([2.0 * (q_y * q_w - q_x * q_z) - accel_measurement[1],
                          2.0 * (q_x * q_y + q_z * q_w) - accel_measurement[2],
                          2*(0.5 - q_y**2 - q_z**2) - accel_measurement[3]]).T# 3x1
            
        J = np.array([[-2*q_z, 2*q_w, -2*q_x, 2*q_y],
                        [2*q_y, 2*q_x, 2*q_w, 2*q_z],
                        [0, -4*q_y, -4*q_z, 0]]) # 3x4
        
        delta_f = np.dot(J.T, f) # 4x1

        delta_q_accel = - beta * (delta_f/np.linalg.norm(f)) # 4x1

        delta_q_gyro = 0.5 * quat_product(q_xyzw,gyro_measurement.T)

        q_dot = delta_q_gyro + delta_q_accel

        self.current_state["q_xyzw"] = q_xyzw + q_dot*self.imu_measurement["dt"]

        return