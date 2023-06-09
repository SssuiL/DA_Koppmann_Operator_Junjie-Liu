import numpy as np
import airsim_para as para
from scipy.spatial.transform import Rotation
import airsim


def qdot(q1, q2):
    q1 = q1.reshape(q1.size, 1)
    q2 = q2.reshape(q2.size, 1)
    q1w, q1v = q1[0, 0], q1[1:4]
    q2w, q2v = q2[0, 0], q2[1:4]
    qout_w = q1w * q2w - q1v.T @ q2v  # the result is a 2-D np array
    qout_v = q1w * q2v + q2w * q1v + np.cross(q1v, q2v, axisa=0, axisb=0).T
    qout = np.concatenate((qout_w, qout_v), axis=0)
    return qout


def quat2rotm(q):
    q = q.reshape(q.size, 1)
    w, x, y, z = q[0, 0], q[1, 0], q[2, 0], q[3, 0]
    rot_matrix = np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]])
    return rot_matrix


def EulerZYX2Quat(eul):
    eul = eul.reshape(eul.size, 1)
    phi, theta, psi = eul[0, 0], eul[1, 0], eul[2, 0]
    qw = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    qx = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    qy = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    qz = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    q = np.array([[qw, qx, qy, qz]]).T
    return q


def Quat2EulerZYX(q):
    q = q.reshape(q.size, 1)
    qw, qx, qy, qz = q[0, 0], q[1, 0], q[2, 0], q[3, 0]
    phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    theta = -np.pi / 2 + 2 * np.arctan2((1 + 2 * (qw * qy - qx * qz)) ** 0.5, (1 - 2 * (qw * qy - qx * qz)) ** 0.5)
    psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    eul = np.array([[phi, theta, psi]]).T
    return eul


def uav_dynamics_discrete(x_in, u_in, ts):
    x = np.reshape(x_in, (x_in.size, 1))
    u = np.reshape(u_in, (u_in.size, 1))
    u_pseudo = para.rforce2pseudo @ u
    ft = u_pseudo[0, 0]
    tau = u_pseudo[1:4]
    N_p, N_v, q, B_w = x[0:3], x[3:6], x[6:10], x[10:13]
    q_conj = q * np.array([[1, -1, -1, -1]]).T
    B_ft_q = np.array([[0, 0, 0, ft]], dtype=np.float64)
    N_ft_q = qdot(qdot(q, B_ft_q), q_conj)
    N_ft = N_ft_q[1:4]
    dN_v = N_ft / para.m + np.array([[0, 0, -9.81]]).T
    N_p_next = N_p + N_v*ts
    N_v_next = N_v + dN_v*ts
    delta_q = EulerZYX2Quat(B_w*ts)
    q_next = qdot(q, delta_q)
    q_next = q_next/np.linalg.norm(q_next)
    dB_w = para.inertia_inv @ tau - para.inertia_inv @ np.cross(B_w, para.inertia_matrix @ B_w, axisa=0, axisb=0).T
    B_w_next = B_w + dB_w*ts
    x_next = np.concatenate((N_p_next, N_v_next, q_next, B_w_next), axis=0)
    return x_next


def uav_attitude_dynamics(xin, uin):
    # 1. Extract info
    x = np.reshape(xin, (xin.size, 1))
    phi, theta, psi = x[0, 0], x[1, 0], x[2, 0]
    Bw = x[3:]
    tau = np.reshape(uin, (uin.size, 1))
    # 2. Derive xdot
    T = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
    deuler = T @ Bw
    dBw = para.inertia_inv @ tau - para.inertia_inv @ np.cross(Bw, para.inertia_matrix @ Bw, axisa=0, axisb=0).T
    xdot = np.vstack((deuler, dBw))
    return xdot


def dfbc(xin, uin, xflat):
    # xin = [p, v, q, Bw], uin = [F1, F2, F3, F4], xflat = 4x5 nparray

    # step 1: get the current & reference state
    xin = xin.reshape(xin.size, 1)
    uin = uin.reshape(uin.size, 1)
    Npr, Nvr, Nar, Ndar, Nddar = xflat[0:3, 0, np.newaxis], xflat[0:3, 1, np.newaxis], xflat[0:3, 2, np.newaxis], \
        xflat[0:3, 3, np.newaxis], xflat[0:3, 4, np.newaxis]
    psir, dpsir, ddpsir = xflat[3, 0], xflat[3, 1], xflat[3, 2]
    Np, Nv, q, Bw = xin[0:3], xin[3:6], xin[6:10], xin[10:13]
    q_conj = q * np.array([[1, -1, -1, -1]]).T

    # step 2: derive fd --> u_pse1, zbd
    ad = para.Kp @ (Npr - Np) + para.Kv @ (Nvr - Nv) + para.Ka @ Nar  # ad = fd/m + [0,0,-g]
    if ad[2, 0] < -7:
        ad = ad * (-7)/ad[2, 0]
    fd = (ad - np.array([[0, 0, -9.81]]).T) * para.m
    zbt_q = qdot(qdot(q, np.array([[0, 0, 0, 1.0]]).T), q_conj)
    zbt = zbt_q[1:4]
    u_pse1 = zbt.T @ fd  # upse1 is a 2-D np array with 1 element
    u_pse1 = np.maximum(0.5, u_pse1)
    if np.linalg.norm(fd) == 0:
        zbd = zbt
    else:
        zbd = fd / np.linalg.norm(fd)

    # step 3: derive the desired orientation qd
    yc = np.array([[-np.sin(psir), np.cos(psir), 0]]).T
    xbd_dirction = np.cross(yc, zbd, axisa=0, axisb=0).T
    xbd = xbd_dirction / np.linalg.norm(xbd_dirction)
    ybd = np.cross(zbd, xbd, axisa=0, axisb=0).T
    Rd = np.concatenate((xbd, ybd, zbd), axis=1)
    qd_xyzw = Rotation.from_matrix(Rd).as_quat()  # this is a 1-D np array with q = [qx, qy, qz, qw]
    qd = np.roll(qd_xyzw, shift=1)  # right shift operator
    qd = qd.reshape(qd.size, 1)

    # step 4: derive hw --> Bwd
    Nat = u_pse1 * zbt / para.m + np.array([[0, 0, -9.81]]).T
    Ndad = para.Kp @ (Nvr - Nv) + para.Kv @ (Nar - Nat) + para.Ka @ Ndar

    d_norm_fd = zbd.T @ Ndad * para.m  # option 1
    # d_norm_fd = zbt.T@Ndad*para.m  # option 2

    hw = (para.m * Ndad - d_norm_fd * zbd) / np.linalg.norm(fd)  # option 1
    # hw = (para.m*Ndad - d_norm_fd*zbt)/np.sum(uin)  # option 2

    Bwxd = -hw.T @ ybd  # option 1
    Bwyd = hw.T @ xbd  # option 1
    Bwzd = dpsir * np.array([[0, 0, 1.0]]) @ zbd  # option 1
    # Rt = quat2rotm(q)  # option 2
    # Bwxd = -hw.T @ Rt[:, 1, np.newaxis]  # option 2
    # Bwyd = hw.T @ Rt[:, 0, np.newaxis]  # option 2
    # Bwzd = dpsir * np.array([[0, 0, 1.0]]) @ Rt[:, 2, np.newaxis]  # option 2
    Bwd = np.concatenate((Bwxd, Bwyd, Bwzd), axis=0)

    # step 5: orientation control
    qe = qdot(qd, q_conj)
    qew, qex, qey, qez = qe[0, 0], qe[1, 0], qe[2, 0], qe[3, 0]
    qe_red = 1 / np.sqrt(qew ** 2 + qez ** 2) * np.array([[qew * qex - qey * qez, qew * qey + qex * qez, 0]]).T
    qe_yaw = 1 / np.sqrt(qew ** 2 + qez ** 2) * np.array([[0, 0, qez]]).T
    dBwd = para.Kq @ qe_red + np.sign(qew) * para.Kq @ qe_yaw + para.Kw @ (Bwd - Bw)

    # step 6: derive u_pseudo & u_thrust
    taud = para.inertia_matrix @ dBwd + np.cross(Bw, para.inertia_matrix @ Bw, axisa=0, axisb=0).T
    u_pseudo = np.concatenate((u_pse1, taud), axis=0)
    u_thrust = para.rforce2pseudo_inv @ u_pseudo
    return u_thrust


def generate_circular_xflat(ts, t_total, r, w, h):
    time = np.arange(0, t_total + ts, ts)
    pref, vref, aref, daref, ddaref, psiref = np.zeros((time.size, 3)), np.zeros((time.size, 3)), \
        np.zeros((time.size, 3)), np.zeros((time.size, 3)), np.zeros((time.size, 3)), np.zeros((time.size, 5))
    for i in range(time.size):
        tc = time[i]
        px, py, pz = -r * np.sin(w*tc), r * np.cos(w*tc), h
        vx, vy, vz = -w * r * np.cos(w*tc), -w * r * np.sin(w*tc), 0
        ax, ay, az = w**2 * r * np.sin(w*tc), -w**2 * r * np.cos(w*tc), 0
        dax, day, daz = w**3 * r * np.cos(w*tc), w**3 * r * np.sin(w*tc), 0
        ddax, dday, ddaz = -w**4 * r * np.sin(w*tc), w**4 * r * np.cos(w*tc), 0
        pref[i] = np.array([px, py, pz])
        vref[i] = np.array([vx, vy, vz])
        aref[i] = np.array([ax, ay, az])
        daref[i] = np.array([dax, day, daz])
        ddaref[i] = np.array([ddax, dday, ddaz])
    return pref, vref, aref, daref, ddaref, psiref


def generate_realT_circular_xflat(tc, r, w, h):
    px, py, pz = -r * np.sin(w * tc), r * np.cos(w * tc), h
    vx, vy, vz = -w * r * np.cos(w * tc), -w * r * np.sin(w * tc), 0
    ax, ay, az = w ** 2 * r * np.sin(w * tc), -w ** 2 * r * np.cos(w * tc), 0
    dax, day, daz = w ** 3 * r * np.cos(w * tc), w ** 3 * r * np.sin(w * tc), 0
    ddax, dday, ddaz = -w ** 4 * r * np.sin(w * tc), w ** 4 * r * np.cos(w * tc), 0
    xflat_ref = np.array([[px, vx, ax, dax, ddax],
                          [py, vy, ay, day, dday],
                          [pz, vz, az, daz, ddaz],
                          [0, 0, 0, 0, 0]])
    return xflat_ref


def poly_derivative(tc):
    M = np.array([[1, tc, tc**2, tc**3, tc**4, tc**5, tc**6, tc**7],
                  [0, 1, 2*tc, 3*tc**2, 4*tc**3, 5*tc**4, 6*tc**5, 7*tc**6],
                  [0, 0, 2, 6*tc, 12*tc**2, 20*tc**3, 30*tc**4, 42*tc**5],
                  [0, 0, 0, 6, 24*tc, 60*tc**2, 120*tc**3, 210*tc**4],
                  [0, 0, 0, 0, 24, 120*tc, 360*tc**2, 840*tc**3],
                  [0, 0, 0, 0, 0, 120, 720*tc, 2520*tc**2],
                  [0, 0, 0, 0, 0, 0, 720, 5040*tc]])
    return M


def BIVP_snap(waypoints, v_tips, a_tips, time_stamp):
    #  waypoints should be an N x 3 numpy array (2D). time_stamp is a 1-D numpy array
    time_interval = time_stamp[1:time_stamp.size] - time_stamp[0:-1]
    M = np.zeros((8*time_interval.size, 8*time_interval.size))
    temp1 = np.diag([1.0, 1.0, 2.0, 6.0])
    temp2 = np.zeros((4, 4))
    F0 = np.concatenate((temp1, temp2), 1)
    tM = time_interval[-1]
    derivative_matrix = poly_derivative(tM)
    EM = derivative_matrix[:4]
    M[:4, :8] = F0
    M[-4:, -8:] = EM
    Fi = np.diag([-1.0, -1.0, -2.0, -6.0, -24.0, -120.0, -720.0], -1)
    b = np.zeros((8*time_interval.size, 3))
    b[0] = waypoints[0]
    b[1] = v_tips[0]
    # b[2] = a_tips[0]
    b[-4] = waypoints[-1]
    b[-3] = v_tips[1]
    # b[-2] = a_tips[1]
    for i in range(time_interval.size-1):
        temp = poly_derivative(time_interval[i])
        Ei = np.concatenate((temp[0, np.newaxis], temp), 0)
        M[4+8*i:4+8*(i+1), 8*i:8*i+16] = np.concatenate((Ei, Fi), 1)
        b[4+8*i] = waypoints[i+1]
    C = np.linalg.inv(M) @ b
    return C


def Generate_xflat_ref(C, time_stamp, ts):
    time = np.arange(0, time_stamp[-1]+ts, ts)
    xflat_pref, xflat_vref, xflat_aref, xflat_jref, xflat_sref = np.zeros((time.size, 3)), np.zeros((time.size, 3)), \
        np.zeros((time.size, 3)), np.zeros((time.size, 3)), np.zeros((time.size, 3))
    for i in range(time.size):
        t_current = time[i]
        seg_idx = 0
        for j in range(time_stamp.size-1):
            if t_current <= time_stamp[j+1]:
                seg_idx = j
                break
        C_select = C[8*seg_idx:8*(seg_idx+1)]
        t_delta = t_current - time_stamp[seg_idx]
        derivative_matrix = poly_derivative(t_delta)
        BIVP_info = derivative_matrix @ C_select
        xflat_pref[i], xflat_vref[i], xflat_aref[i], xflat_jref[i], xflat_sref[i] = BIVP_info[0], BIVP_info[1], \
            BIVP_info[2], BIVP_info[3], BIVP_info[4]
    return xflat_pref, xflat_vref, xflat_aref, xflat_jref, xflat_sref


def generate_realT_BIVP_xflat(C, time_stamp, tc):
    seg_idx = 0
    for i in range(time_stamp.size-1):
        if tc <= time_stamp[i+1]:
            seg_idx = i
            break
    C_select = C[8 * seg_idx:8 * (seg_idx + 1)]
    t_delta = tc - time_stamp[seg_idx]
    derivative_matrix = poly_derivative(t_delta)
    BIVP_info = derivative_matrix @ C_select
    xflat = np.zeros((4, 5))
    xflat[:3] = BIVP_info[:5].T
    return xflat


def extract_airsim_state(state_ak):  # state_ak = airsim kinematic state, change NED to NEU
    Np = np.array([[-state_ak.position.x_val, state_ak.position.y_val, -state_ak.position.z_val]]).T
    Nv = np.array([[-state_ak.linear_velocity.x_val,
                    state_ak.linear_velocity.y_val,
                    -state_ak.linear_velocity.z_val]]).T
    (pitch, roll, yaw) = airsim.to_eularian_angles(state_ak.orientation)
    q = EulerZYX2Quat(np.array([[-roll, pitch, -yaw]]).T)
    Bw = np.array([[-state_ak.angular_velocity.x_val, state_ak.angular_velocity.y_val,
                    -state_ak.angular_velocity.z_val]]).T
    quad_state = np.vstack((Np, Nv, q, Bw))
    return quad_state


def extract_airsim_thrust(rotor):
    rotor_state = rotor.rotors
    thrust = np.array([[rotor_state[0]["thrust"], rotor_state[1]["thrust"], rotor_state[2]["thrust"],
                        rotor_state[3]["thrust"]]]).T
    return thrust
