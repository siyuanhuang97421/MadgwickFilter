from DataIterator import *
from DataLoader import *

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# from helper.rotplot import rotplot
import time
from pyquaternion import Quaternion
from Filter import MadgwickFilter

def wrap_angle(prev,cur):
    for i in range(cur.shape[0]):
        if abs(prev[i]-cur[i])>6.0:
            if cur[i]>6.0:
                cur[i]= cur[i]-2*np.pi
            else:
                cur[i] = cur[i] + 2*np.pi
    return cur

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

    # q1 = Quaternion(q1_w, q1_x, q1_y, q1_z)
    # q2 = Quaternion(q2_w, q2_x, q2_y, q2_z)

    # result = q1 * q2
    
    # return np.array([result[1],result[2],result[3],result[0]])

def quaternion_to_euler(q):
    rm = quaternion_to_RM(q)

    # q_x = q[0]
    # q_y = q[1]
    # q_z = q[2]
    # q_w = q[3]

    # roll = np.arctan2(2*q_x*q_y - 2*q_w*q_z, 2*(q_w**2 + q_x**2) - 1)
    # pitch = -1*np.arcsin(2*(q_x * q_z + q_w * q_y))
    # yaw = np.arctan2(q_y*q_z + 2*q_w*q_x, 2*(q_w**2 + q_z**2) - 1)


    # Convert rotation matrix to Rotation object
    rotation = R.from_matrix(rm)

    # Convert Rotation object to Euler angles
    euler_angles = rotation.as_euler('zyx', degrees=False)

    # print("Euler angles (yaw, pitch, roll):", euler_angles)

    # rot = R.from_quat(q)
    # euler = rot.as_euler('ZYX')

    roll = euler_angles[2]
    pitch = euler_angles[1]
    yaw = euler_angles[0]

    # quaternion = Quaternion(q[3], q[0], q[1], q[2])  # Create a Quaternion object (w,x,y,z)
    # euler_angles = quaternion.euler_angles('zyx')  # Convert to Euler angles
    return roll, pitch, yaw

def quaternion_to_RM(q):
    quaternion = Quaternion(q[3], q[0], q[1], q[2])  # Create a Quaternion object (w,x,y,z)
    rm = quaternion.rotation_matrix # Convert to rotation matrix
    return rm

def RM_to_quaternion(rm):
    q = Quaternion(matrix=rm) # w,x,y,z
    return np.array([q[1], q[2], q[3], q[0]]).T # x,y,z,w

class Phase1:
    def __init__(self,imu_data, vicon_data):
        # self.imu_params = io.loadmat(imu_params_path)
        # self.imu_data = io.loadmat(imu_data_path)
        # self.vicon_data = io.loadmat(vicon_data_path)

        # self.acc_scale = self.imu_params['IMUParams'][0]
        # self.acc_bias = self.imu_params['IMUParams'][1]

        # self.ts_imu = self.imu_data['ts']
        # # print(len(self.ts_imu[0]))

        # a_x = self.imu_data['vals'][0]
        # a_y = self.imu_data['vals'][1]
        # a_z = self.imu_data['vals'][2]
        # w_z = self.imu_data['vals'][3]
        # w_x = self.imu_data['vals'][4]
        # w_y = self.imu_data['vals'][5]

        # self.a_xbar = (a_x + self.acc_bias[0])/self.acc_scale[0]
        # self.a_ybar = (a_y + self.acc_bias[1])/self.acc_scale[1]
        # self.a_zbar = (a_z + self.acc_bias[2])/self.acc_scale[2]


        # self.a_xbar = a_x*self.acc_scale[0]+self.acc_bias[0]
        # self.a_ybar = a_y*self.acc_scale[1]+self.acc_bias[1]
        # self.a_zbar = a_z*self.acc_scale[2]+self.acc_bias[2]

        # self.w_xbar = (3300/1023)*(np.pi/180)*0.3*(w_x - np.mean(w_x[:100]))
        # # print(w_xbar)
        # self.w_ybar = (3300/1023)*(np.pi/180)*0.3*(w_y - np.mean(w_y[:100]))
        # # print(w_ybar)
        # self.w_zbar = (3300/1023)*(np.pi/180)*0.3*(w_z - np.mean(w_z[:100]))
        # # print(w_zbar)
        self.imu_data = imu_data
        self.vicon_data = vicon_data

        self.yaw_values_vicon = []
        self.pitch_values_vicon = []
        self.roll_values_vicon = []
        self.time_values_vicon = []

        self.roll_accl = []
        self.pitch_accl = []
        self.yaw_accl = []
        self.time_values_accl = []

        self.roll_gyro = []
        self.pitch_gyro = []
        self.yaw_gyro = []
        self.time_values_gyro = [] 


        self.cf_roll=[]
        self.cf_pitch=[]
        self.cf_yaw=[]

        self.gyro_rotation_matrices=[]
        self.accl_rotation_matrices=[]
        self.cf_rotation_matrices=[]

        self.q=[]
        self.roll_madg=[]
        self.pitch_madg=[]
        self.yaw_madg=[]

        # self.rots = self.vicon_data['rots']
        # self.ts_vicon = self.vicon_data['ts']
        # print(len(self.ts_vicon[0]))


        self.start_index_imu = 0
        self.start_index_vicon = 0
        

    def load_vicon_data(self):
        prev_euler_angles = None
        vicon_itr = ViconDataIterator(self.vicon_data)
        while True:
            vicon_measurement = vicon_itr.getCurrentData()

            rotation_matrix = vicon_measurement["rot_matrix"]

            rotation = R.from_matrix(rotation_matrix)

            euler_angles = rotation.as_euler('ZYX', degrees=False)

            # if prev_euler_angles is not None:
            #     euler_angles = wrap_angle(prev_euler_angles,euler_angles)
            # prev_euler_angles=euler_angles

            self.yaw_values_vicon.append(euler_angles[0])
            self.pitch_values_vicon.append(euler_angles[1])
            self.roll_values_vicon.append(euler_angles[2])
            
            self.time_values_vicon.append(vicon_measurement["t"])

            if not vicon_itr.step():
                break


    def gyro(self):
        initial_roll = self.roll_values_vicon[0]  
        initial_pitch = self.pitch_values_vicon[0] 
        initial_yaw = self.yaw_values_vicon[0]  

        self.roll_gyro = [initial_roll]
        self.pitch_gyro = [initial_pitch]
        self.yaw_gyro = [initial_yaw]
        self.time_values_gyro = [self.ts_imu[0][0]]  

        for i in range(1, len(self.ts_imu[0])):
            delta_time = self.ts_imu[0][i] - self.ts_imu[0][i - 1]

            # delta_roll = (self.w_xbar[i - 1]+self.w_xbar[i ])*0.5 * delta_time
            # delta_pitch = (self.w_ybar[i - 1]+self.w_ybar[i ])*0.5 * delta_time
            # delta_yaw = (self.w_zbar[i - 1]+self.w_zbar[i ])*0.5 * delta_time

            current_rotation = R.from_euler('zyx',[self.yaw_gyro[-1],self.pitch_gyro[-1],self.roll_gyro[-1]],degrees=False).as_matrix()
            current_gyro_data = np.array([self.w_xbar[i - 1],self.w_ybar[i - 1],self.w_zbar[i - 1]]).reshape([3,1])
            delta_rotation = np.matmul(current_rotation,current_gyro_data).T*delta_time
            delta_roll = delta_rotation[0,0]
            delta_pitch = delta_rotation[0,1]
            delta_yaw = delta_rotation[0,2]

            # delta_roll = self.w_xbar[i - 1] * delta_time
            # delta_pitch = self.w_ybar[i - 1] * delta_time
            # delta_yaw = self.w_zbar[i - 1] * delta_time

            new_roll = self.roll_gyro[-1] + delta_roll
            new_pitch = self.pitch_gyro[-1] + delta_pitch
            new_yaw = self.yaw_gyro[-1] + delta_yaw

            self.roll_gyro.append(new_roll)
            self.pitch_gyro.append(new_pitch)
            self.yaw_gyro.append(new_yaw)

            self.time_values_gyro.append(self.ts_imu[0][i])

    def accl(self):
        for i in range(len(self.ts_imu[0])):
            new_roll = np.arctan2(self.a_ybar[i], np.sqrt(self.a_xbar[i]**2 + self.a_zbar[i]**2))
            new_pitch = np.arctan2(-1*self.a_xbar[i], np.sqrt(self.a_ybar[i]**2 + self.a_zbar[i]**2))
            new_yaw = np.arctan2(np.sqrt(self.a_xbar[i]**2 + self.a_ybar[i]**2), self.a_zbar[i])

            self.roll_accl.append(new_roll)
            self.pitch_accl.append(new_pitch)
            self.yaw_accl.append(new_yaw)

            self.time_values_accl.append(self.ts_imu[0][i])

    def com_filter(self):
        self.filtered_roll_accl = [self.roll_accl[0]]
        self.filtered_pitch_accl = [self.pitch_accl[0]]
        self.filtered_yaw_accl = [self.yaw_accl[0]]

        self.filtered_roll_gyro = [self.roll_gyro[0]]
        self.filtered_pitch_gyro = [self.pitch_gyro[0]]
        self.filtered_yaw_gyro = [self.yaw_gyro[0]]

        alpha_lp = 0.2
        alpha_hp = 0.9999
        alpha_cf = 0.5

        for i in range(1, len(self.ts_imu[0])):
            self.filtered_roll_accl.append((1-alpha_lp)*self.filtered_roll_accl[-1] + alpha_lp*self.roll_accl[i])
            self.filtered_pitch_accl.append((1-alpha_lp)*self.filtered_pitch_accl[-1] + alpha_lp*self.pitch_accl[i])
            self.filtered_yaw_accl.append((1-alpha_lp)*self.filtered_yaw_accl[-1] + alpha_lp*self.yaw_accl[i])

            self.filtered_roll_gyro.append((alpha_hp)*(self.filtered_roll_gyro[-1] + self.roll_gyro[i] - self.roll_gyro[i-1]))
            self.filtered_pitch_gyro.append((alpha_hp)*(self.filtered_pitch_gyro[-1] + self.pitch_gyro[i] - self.pitch_gyro[i-1]))
            self.filtered_yaw_gyro.append((alpha_hp)*(self.filtered_yaw_gyro[-1] + self.yaw_gyro[i] - self.yaw_gyro[i-1]))

        for i in range(len(self.ts_imu[0])):
            self.cf_roll.append((1-alpha_cf)*self.filtered_roll_gyro[i] + alpha_cf*self.filtered_roll_accl[i])
            self.cf_pitch.append((1-alpha_cf)*self.filtered_pitch_gyro[i] + alpha_cf*self.filtered_pitch_accl[i])
            self.cf_yaw.append((1-alpha_cf)*self.filtered_yaw_gyro[i] + alpha_cf*self.filtered_yaw_accl[i])

    def madgwick(self):

        beta = 0.1  # Beta parameter for Madgwick filter

        # initial_q = np.array([0.0, 0.0, 0.0, 1.0]).T  # Initial quaternion (x, y, z, w)
        initial_q = RM_to_quaternion(self.vicon_data.rot_matrices[:,:,0])
        initial_state = {"q_xyzw": initial_q}
        self.q = [initial_q]

        roll, pitch, yaw = quaternion_to_euler(initial_q)
        self.roll_madg = [roll]
        self.pitch_madg = [pitch]
        self.yaw_madg = [yaw]

        imu_data_itr = IMUDataIterator(self.imu_data)
        madgwick_filter = MadgwickFilter(initial_state)
        prev_euler_angles = None
        while imu_data_itr.step():
            imu_measurement = imu_data_itr.getCurrentData()

            madgwick_filter.updateIMUMeasurement(imu_measurement)
            madgwick_filter.step()

            state = madgwick_filter.getCurrentState()

            self.q.append(state["q_xyzw"])
            
            roll, pitch, yaw = quaternion_to_euler(state["q_xyzw"])
            self.roll_madg.append(roll)
            self.pitch_madg.append(pitch)
            self.yaw_madg.append(yaw)

        # print(len(self.roll_madg))


    def plot(self):
        plt.figure(figsize=(20, 15))

        plt.subplot(3,1,1)

        
        
        plt.plot(self.vicon_data.ts, self.roll_values_vicon, label='X Orientation - Vicon',color='k')
        # plt.plot(self.time_values_accl, self.roll_accl, label='X Orientation - Accel',color='b')
        # plt.plot(self.time_values_gyro, self.roll_gyro, label='X Orientation - gyro',color='r')
        # plt.plot(self.time_values_gyro, self.filtered_roll_accl, label='X Orientation - lp filter accl',color='g')
        # plt.plot(self.time_values_gyro, self.filtered_roll_gyro, label='X Orientation - hp filter gyro',color='m')
        # plt.plot(self.time_values_gyro, self.cf_roll, label='X Orientation - comp filter',color='m')

        plt.plot(self.imu_data.ts, self.yaw_madg, label='X Orientation - madg filter',color='g')

        plt.xlabel('Time')
        plt.ylabel('Roll Orientation')
        plt.title('Roll Orientation Over Time')
        plt.legend()

        # plt.figure(figsize=(10, 6))
        plt.subplot(3,1,2)
        plt.plot(self.vicon_data.ts, self.pitch_values_vicon, label='Y Orientation - Vicon',color='k')
        # plt.plot(self.time_values_accl, self.pitch_accl, label='Y Orientation - Accel',color='b')
        # plt.plot(self.time_values_gyro, self.pitch_gyro, label='Y Orientation - gyro',color='r')
        # plt.plot(self.time_values_gyro, self.filtered_pitch_accl, label='Y Orientation - lp filter accl',color='g')
        # plt.plot(self.time_values_gyro, self.filtered_pitch_gyro, label='Y Orientation - hp filter gyro',color='m')
        # plt.plot(self.time_values_gyro, self.cf_pitch, label='Y Orientation - comp filter',color='m')

        plt.plot(self.imu_data.ts, self.pitch_madg, label='Y Orientation - madg filter',color='g')

        plt.xlabel('Time')
        plt.ylabel('Pitch Orientation')
        plt.title('Pitch Orientation Over Time')
        plt.legend()

        # plt.figure(figsize=(10, 6))
        plt.subplot(3,1,3)
        plt.plot(self.vicon_data.ts, self.yaw_values_vicon, label='Z Orientation - Vicon',color='k')
        # plt.plot(self.time_values_accl, self.yaw_accl, label='Z Orientation - Accel',color='b')
        # plt.plot(self.time_values_gyro, self.yaw_gyro, label='Z Orientation - gyro',color='r')
        # plt.plot(self.time_values_gyro, self.filtered_yaw_accl, label='Z Orientation - lp filter accl',color='g')
        # plt.plot(self.time_values_gyro, self.filtered_yaw_gyro, label='Z Orientation - hp filter gyro',color='m')
        # plt.plot(self.imu_data.ts, self.cf_yaw, label='Z Orientation - comp filter',color='m')

        plt.plot(self.imu_data.ts, self.roll_madg, label='Z Orientation - madg filter',color='g')

        plt.xlabel('Time')
        plt.ylabel('Yaw Orientation')
        plt.legend()
        plt.title('Yaw Orientation Over Time')

        plt.show()
    
    def rotplotter(self):
        fig = plt.figure()

        myAxis1 = fig.add_subplot(151, projection='3d')
        myAxis2 = fig.add_subplot(152, projection="3d")
        myAxis3 = fig.add_subplot(153, projection="3d")
        myAxis4 = fig.add_subplot(154, projection="3d")
        myAxis5 = fig.add_subplot(155, projection="3d")

        for i in range(0,self.length,20):
            imu_idx = i + self.start_index_imu
            vicon_idx = i + self.start_index_vicon

            rotation_gyro = R.from_euler('zyx', [self.yaw_gyro[imu_idx], self.pitch_gyro[imu_idx], self.roll_gyro[imu_idx]], degrees=False)
            rotation_matrix_gyro = rotation_gyro.as_matrix()
            # self.gyro_rotation_matrices.append(rotation_matrix_gyro)

            rotation_accl = R.from_euler('zyx', [self.yaw_accl[imu_idx], self.pitch_accl[imu_idx], self.roll_accl[imu_idx]], degrees=False)
            rotation_matrix_accl = rotation_accl.as_matrix()
            # self.accl_rotation_matrices.append(rotation_matrix_accl)

            rotation_cf = R.from_euler('zyx', [self.cf_yaw[imu_idx], self.cf_pitch[imu_idx], self.cf_roll[imu_idx]], degrees=False)
            rotation_matrix_cf = rotation_cf.as_matrix()
            # self.cf_rotation_matrices.append(rotation_matrix_cf)

            rotation_matrix_madg = quaternion_to_RM(self.q[imu_idx])

            rotation_matrix_vicon = self.rots[:, :, vicon_idx]

            myAxis1.clear()
            rotplot(rotation_matrix_gyro, myAxis1)
            plt.sca(myAxis1)
            plt.title("Gyroscope")

            myAxis2.clear()
            rotplot(rotation_matrix_accl, myAxis2)
            plt.sca(myAxis2)
            plt.title("Accelerometer")

            myAxis3.clear()
            rotplot(rotation_matrix_cf, myAxis3)
            plt.sca(myAxis3)
            plt.title("Complementary")

            myAxis4.clear()
            rotplot(rotation_matrix_madg, myAxis4)
            plt.sca(myAxis4)
            plt.title("Madgwick")

            myAxis5.clear()
            rotplot(rotation_matrix_vicon, myAxis5)
            plt.sca(myAxis5)
            plt.title("Vicon")

            plt.pause(0.005)

            # print(i)
        plt.show()
        


def main():

    trial_num = 1
    # imu_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/Data/Train/IMU/imuRaw{}.mat".format(trial_num)
    imu_path = "Data/Train/IMU/imuRaw{}.mat".format(trial_num)
    imu_data_loader = MatDataLoader(imu_path)
    imu_data = IMUDataset(imu_data_loader)

    # vicon_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/Data/Train/Vicon/viconRot{}.mat".format(trial_num)
    vicon_path = "Data/Train/Vicon/viconRot{}.mat".format(trial_num)
    vicon_data_loader = MatDataLoader(vicon_path)
    vicon_data = ViconDataset(vicon_data_loader)

    # accel_param_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/IMUParams.mat"
    accel_param_path = "IMUParams.mat"
    accel_param_loader = MatDataLoader(accel_param_path)
    accel_param = AccelParam(accel_param_loader)

    imu_idx, vicon_idx = findCommonStartTimeIdx(imu_data.ts, vicon_data.ts)
    imu_data.removeDataBeforeIdx(imu_idx)
    vicon_data.removeDataBeforeIdx(vicon_idx)
    imu_data.correctBias(accel_param)

    # imu_params_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/IMUParams.mat"
    # imu_data_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/Data/Train/IMU/imuRaw1.mat"
    # vicon_data_path = "/home/mayank/Desktop/Drones/YourDirectoryID_p0_dev/Phase1/Data/Train/Vicon/viconRot1.mat"    

    phase1 = Phase1(imu_data, vicon_data)

    phase1.load_vicon_data()
    # phase1.gyro()
    # phase1.accl()
    # phase1.com_filter()
    phase1.madgwick()
    # phase1.sync_time()
    phase1.plot()
    # phase1.rotplotter()

if __name__ == "__main__":
    main()