from time import time, sleep
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np

def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # ZYX (yaw-pitch-roll) 순서
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    # R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

# --------- desired ---------
desired_xpos_tcp = np.array([0.0, -0.5, 0.1])
desired_rpy = np.array([90.0, 0.0, 0.0])  # roll, pitch, yaw 입력
# -----------------------------

model_path = "/home/chu/manipulator_control/rb5/scene_rb5_gripper.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

d.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
d.qpos[6:] = [0.010, 0.010]   # 초기 gripper open

M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

C0 = np.zeros((6,6))

K_a = np.array([50.0, 50.0, 35.0])
zeta_a = np.array([9.0, 9.0, 4.0])

K_o = np.array([3.0, 3.0, 1.5])
zeta_o = np.array([0.3, 0.3, 0.2])

d.ctrl[6] = 0.012   # open
#d.ctrl[6] = 0.000   # close

with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time()
    while viewer.is_running():
        t = time() - t0

        
        mujoco.mj_fullM(m, M, d.qM)

        qvel_backup = deepcopy(d.qvel)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G) 
        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)        

        np.fill_diagonal(C0, np.sum(np.abs(M[0:6, 0:6]), axis=1))

        tcp_site_id = m.site("tcp").id
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        jacp0 = deepcopy(jacp[:, 0:6])
        jacr0 = deepcopy(jacr[:, 0:6])
        # print(f"jr : {jacr0}")

        # Position Error
        xpos_err0 = d.site("tcp").xpos - desired_xpos_tcp

        # Orientation Error (RPY 입력 반영)
        R_current = d.site(tcp_site_id).xmat.reshape(3, 3)
        # print(f"Ro : {R_current}")
        
        R_desired = rpy_to_rotmat(*desired_rpy)
        # print(f"Rd : {R_desired}")
        # R_desired = R_desired@R_desired
        ori_err0 = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        ) # e  = (x X xd) + (y X yd) + (z X zd)
        
        # Angular Velocity
        w0 = jacr0 @ d.qvel[0:6]
        # print(f"wo : {w0}")

        # Orientation Force
        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # Linear Velocity
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        # Linear Force 
        force0  = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

        M0 = M[0:6, 0:6]
        M0_inv = np.linalg.pinv(M0)

        eps = 1e-6
        Lambda_inv = jacp0 @ M0_inv @ jacp0.T
        Lambda_pos = np.linalg.pinv(Lambda_inv + eps * np.eye(3))

        force0_task = Lambda_pos @ force0


        # Torque (PD + Gravity + Damping + Orientation)
        torque0 = ( -0 * C0 @ d.qvel[0:6] 
                    - 1 * jacp0.T @ force0_task
                    + 1 * G[0:6] 
                    - 1 * jacr0.T @ F_ori_0)
        # print(f"torque 0 : {torque0}")
        
        max_torque = 80
        d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
        # d.qvel[:] = 0
        #print(f"Taget torque : {d.ctrl[0:6]}")

        current_tcp_pos = d.site("tcp").xpos
        print(f"Current TCP Position : X={current_tcp_pos[0]:.4f}, Y={current_tcp_pos[1]:.4f}, Z={current_tcp_pos[2]:.4f}")
        #R_err = R_desired.T @ R_current
        #angle_err = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        #print(f"Orientation error [deg]: {np.rad2deg(angle_err):.3f}")
        #print("grasp_center:", d.site("grasp_center").xpos)
        #print("peg_tip_nominal:", d.site("peg_tip_nominal").xpos)

        mujoco.mj_step(m, d)
        viewer.sync()


'''
wrench vector : [ f m ].T

f = kv * ohm * (ph - pp) + f*
    ohm : R33 : generalized task-specification matrix
               (to seperate position control space frome force control space)
    f* : desired force and position control input
    kv : proportional gain for the position control
    ph : hole position
    pp : peg position
    
m = Kw * deltaPi
    deltaPi = Er(R*Rh, Rp) :angular rotation matrix of the peg
    Er(A, B) = [a1 X b1 
               +a2 X b2
               +a3 X b3]
    R* : desired rotation matrix of the peg -> target tcp rmat
    Rh : rotation matrix of the hole -> target tcp rmat
    Rp : rotation matrix of the peg -> current tcp rmat
    
    Kw : orientation control gain matrix (diag(gs,gw,gw))
        gs : control gains screwing motion         
        gw : control gains wiggling motion  
'''



'''
from rbpodo import Cobot
import rbpodo as rb
import numpy as np
import time

ROBOT_ADDRESS = "169.254.186.10"

robot = rb.Cobot(ROBOT_ADDRESS)
rc = rb.ResponseCollector()

robot.set_operation_mode(rc, rb.OperationMode.Real)
robot.set_speed_bar(rc, 1.0)

t1 = 0.01 #이동시간
t2 = 0.1 #유지시간
compensation = 3   # gravity+friction

repeat_count = 20  # 반복 횟수

for i in range(repeat_count):
    for val in [+10, -10, 0]:  # +1 → -1 → 0 반복
        target_torque = np.zeros(6)
        target_torque[5] = val  # 6번 관절만 토크 부여
        ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation)
        if not ret.is_success():
            print(f"move_servo_t 실패 (joint 5, val {val}):", ret)
            break
        time.sleep(5)

'''