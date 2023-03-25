# Load regression data:
import numpy as np


def load_sensor_data(filename='./data/flight_data.csv'):
    # import a csv file:
    data = np.genfromtxt(filename, names=True, delimiter=',')

    # Extract the accelerometer data, gyroscopes, attitude angles, and control commands:
    N = len(data['accel_x'])
    accel_x = np.reshape(data['accel_x'], [N, 1])
    accel_y = np.reshape(data['accel_y'], [N, 1])
    accel_z = np.reshape(data['accel_z'], [N, 1])
    gyro_p = np.reshape(data['gyro_p'], [N, 1])
    gyro_q = np.reshape(data['gyro_q'], [N, 1])
    gyro_r = np.reshape(data['gyro_r'], [N, 1])
    att_phi = np.reshape(data['att_phi'], [N, 1])
    att_theta = np.reshape(data['att_theta'], [N, 1])
    att_psi = np.reshape(data['att_psi'], [N, 1])
    cmd_thrust = np.reshape(data['cmd_thrust'], [N, 1])
    cmd_roll = np.reshape(data['cmd_roll'], [N, 1])
    cmd_pitch = np.reshape(data['cmd_pitch'], [N, 1])
    cmd_yaw = np.reshape(data['cmd_yaw'], [N, 1])

    return [accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r, att_phi, att_theta, att_psi, cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw]


def load_sine_data(a=1, b=1, c=0, d=0, N=1000, noise=0.1):

    # Generate a sine wave:
    t = np.linspace(-np.pi, np.pi, N)
    t = t.reshape(N, 1)
    x = a*np.sin(b*t + c) + d
    y = x + noise*np.random.randn(N, 1)

    return [t, y]
