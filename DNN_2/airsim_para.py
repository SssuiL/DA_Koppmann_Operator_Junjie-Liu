import control
import numpy as np
pi = np.pi


def compute_inertia_matrix(box_x, box_y, box_z, box_mass, motor_w, quad_poses):
    inertia = np.zeros((3, 3))
    inertia[0, 0] = box_mass/12 * (box_y**2 + box_z**2)
    inertia[1, 1] = box_mass/12 * (box_x**2 + box_z**2)
    inertia[2, 2] = box_mass/12 * (box_x**2 + box_y**2)
    for i in range(4):
        inertia[0, 0] += (quad_poses[i, 1]**2 + quad_poses[i, 2]**2) * motor_w
        inertia[1, 1] += (quad_poses[i, 0]**2 + quad_poses[i, 2]**2) * motor_w
        inertia[2, 2] += (quad_poses[i, 0]**2 + quad_poses[i, 1]**2) * motor_w
    return inertia


# Rotor parameters (GWS 9X5, DJI Phantom 2, MT2212 motor)
max_rpm = 6396.667
max_rps = max_rpm/60
c_t = 0.109919  # @ 6396.667 RPM (MAX)
c_p = 0.040164  # @ 6396.667 RPM (MAX)
air_density = 1.225  # kg/m^3
D = 0.2286  # propeller_diameter
max_force = c_t * air_density * D**4 * max_rps**2
max_rotor_torque = c_p * air_density * D**5 * max_rps**2 / (2*pi)
c_tauf = max_rotor_torque / max_force

# Quadrotor parameter (F450 frame, MT2212 motor)
rotor_count = 4
arm_length = 0.2275
m = 1
motor_w = 0.055  # motor assembly weight
box_mass = m - rotor_count * motor_w
box_x, box_y, box_z, real_z = 0.18, 0.11, 0.04, 0.025
align_ang = pi/4
proj_len = arm_length * np.sin(align_ang)
quad_poses = np.array([[proj_len, proj_len, real_z],
                       [-proj_len, -proj_len, real_z],
                       [proj_len, -proj_len, real_z],
                       [-proj_len, proj_len, real_z]])
inertia_matrix = compute_inertia_matrix(box_x, box_y, box_z, box_mass, motor_w, quad_poses)
inertia_inv = np.linalg.inv(inertia_matrix)
rforce2pseudo = np.array([[1, 1, 1, 1],
                          [proj_len, -proj_len, -proj_len, proj_len],
                          [proj_len, -proj_len, proj_len, -proj_len],
                          [-c_tauf, -c_tauf, c_tauf, c_tauf]])
rforce2pseudo_inv = np.linalg.inv(rforce2pseudo)

# DFBC Control Parameters
Kp = np.diag(np.array([10, 10, 10], dtype=np.float64))
Kv = np.diag(np.array([6, 6, 6], dtype=np.float64))
Ka = np.diag(np.array([1, 1, 1], dtype=np.float64))
Kq = np.diag(np.array([190, 190, 3], dtype=np.float64))
Kw = np.diag(np.array([50, 50, 8], dtype=np.float64))
ts = 0.005

# Optimal Control Parameters
x_scale = np.array([16, 16, 16, 20, 20, 20, np.pi, np.pi, 2*np.pi, 20, 20, 20])
u_scale = 4 * np.ones(4)
x_weight = np.array([5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
u_weight = 0.01 * np.ones(4)
opt_scale = np.concatenate((x_scale, u_scale))
opt_weight = np.concatenate((x_weight, u_weight))

# Linearized System
Anom_00, Anom_01, Anom_1 = np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))
Anom = np.vstack((np.hstack((Anom_00, Anom_01)), Anom_1))
Bnom = np.vstack((np.zeros((3, 3)), inertia_inv))
Q = np.diag([2, 2, 2, 0.1, 0.1, 0.1])
R = np.diag([0.5, 0.5, 0.5])
K, S, E = control.lqr(Anom, Bnom, Q, R)

# Following: BIVP
t_layback = 0.5
t_trans = 0.01

# Others
eps_p_norm = 0.02
eps_v_norm = 0.02
eps_psi_norm = 5/180 * np.pi
eps_euler_norm = 0.03  # norm(0.02, 0.02, 0.02) rad
eps_Bw_norm = 0.03  # norm(0.02, 0.02, 0.02) rad/s

