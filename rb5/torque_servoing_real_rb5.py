from time import time, sleep
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb
import numpy as np

######## torque servo
try:
    ROBOT_ADDRESS = "192.169.1.200"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    
    robot_data = CobotData(ROBOT_ADDRESS)
    state = robot_data.request_data()
    
    robot.set_operation_mode(rc, rb.OperationMode.Simulation)
    robot.set_speed_bar(rc, 0.5)
    #robot.set_freedrive_mode(rc, on=False)
    t1 = 0.01 #이동시간
    t2 = 0.05 #유지시간
    
except Exception as e:
    print(f"No Robot Connection ..! {e}")
    pass
########

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

def rb_get_joint_state():
    jpos = []
    jvel = []
    for i in range(6):
        var_pos = getattr(SystemVariable, f"SD_J{i}_ANG")
        var_vel = getattr(SystemVariable, f"SD_J{i}_VEL")
        _, pos = robot.get_system_variable(rc, var_pos)
        _, vel = robot.get_system_variable(rc, var_vel)
        jpos.append(pos)
        jvel.append(vel)
    return np.array(jpos), np.array(jvel)

def rb_get_joint_position():
    jpos = []
    if state is None:
        print("Failed to get robot state.")
    else:
        jpos = state.sdata.jnt_ang  # 조인트 위치 (deg)
    return np.array(jpos)

# --------- desired ---------
desired_xpos_tcp = np.array([0.0, -0.45, 0.35])
desired_rpy = np.array([90.0, 0.0, 0.0])  # roll, pitch, yaw 입력
# -----------------------------

model_path = "/home/kdh/Desktop/delto/delto_description2/rb3_single/scene_rb3.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# initial robot pose
try:
    jpos, jvel = rb_get_joint_state()
    print(f"Initial jpos : {jpos} | jvel : {jvel}")

    while (len(jpos)==0):
        print("Waiting for robot init data..")
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = 0 #TODO
    sleep(3)
except Exception as e:
    print(f"Can't get robot init data ..! {e}")
    d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
    d.qvel[:] = 0
    sleep(3)


M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

C0 = np.zeros((6,6))

K_a = 100.0
zeta_a = 1.0

K_o = 4.0
zeta_o = 1.0

######################
# Friction Coef
Cfc = np.array([10.5, 6.5, 6.5, 0.7, 1.0, 1.5]) # coulomb friction coef
Vfc = np.array([10.5, 6.5, 6.5, 0.7, 1.0, 1.0]) # viscous friction coef
friction_curve_coef = 8*1e-1

######################
prev_time = time()
hz_window = []
prev_jpos = None  

with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time()
    while viewer.is_running():
        state = robot_data.request_data()
        
        if state.sdata.op_stat_collision_occur:
            print("Robot in Collision")
            break
        if state.sdata.op_stat_sos_flag==4:
            print(f"Command Input Error | JVEL : {jvel}")
            break
        
        now = time()
        loop_dt = now - prev_time
        prev_time = now

        mujoco.mj_step(m, d)
        mujoco.mj_fullM(m, M, d.qM)

        try:
        ## real data update ##
            jpos = rb_get_joint_position()
            jvel = np.zeros(6)
            if prev_jpos is not None:
                jvel = (jpos - prev_jpos) / loop_dt
                # print(f"jvel calc :: {jvel} = {jpos}-{prev_jpos}/{loop_dt}")
            prev_jpos = jpos
            
            # print(f"JP : {jpos} | JV : {jvel}")
            
            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)
        except Exception as e:
            print(f"real data update failed..! {e}")
        ######################
        
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
        F_ori_0 = (K_o * ori_err0) #+ (zeta_o * np.sqrt(K_o) * w0)

        # Linear Velocity
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        # Linear Force 
        force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

        # Friction Comp.
        # TODO 
        
        Tf = np.zeros((6))
        mujoco.mj_forward(m, d)
        for i in range(6):
            Tf[i] = (Cfc[i] * np.tanh(friction_curve_coef*d.qvel[i]) + Vfc[i] * d.qvel[i])        
        # print(f"Friction Torque : {Tf}")
        
        # Torque (Coli + Gravity + Damping + Orientation)
        torque0 = (- 0 * C0 @ d.qvel[0:6] 
                   - 0 * np.linalg.pinv(jacp0) @ force0 #j+acp0.T @ force0  # add exp func (prop err)
                   + 1 * G[0:6] 
                   - 0 * jacr0.T @ F_ori_0
                   + 0 * Tf[0:6])
        # print(f"torque 0 : {torque0}")
        
        max_torque = 50
        d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
 
        if np.any(jvel > 70): # Joint Vel Limit
            i = np.where(jvel > 70)[0]  # 튜플에서 실제 인덱스만 가져옴

            if len(i) > 0:
                d.ctrl[i] = 0.0 * torque0[i]
                print(torque0)
                print(f"Joint Velocity is too fast ...! Joint{list(i)} | Jvel : {jvel[i]}")
                
        # print(f"Taget torque : {d.ctrl[0:6]}")
        
        target_torque =  d.ctrl[0:6]

         # Hz 측정 (1 / 주기)
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)
            if len(hz_window) > 30:  # 최근 30프레임 평균
                hz_window.pop(0)
            # print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f}) |\nTaget torque : {d.ctrl[0:6]}")
       
        # 토크 서보잉 입력
        try:
            target_torque =  d.ctrl[0:6]
            ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation=2)
            # # comp 0 : u / 1 : u + g / 2 : u + f / 3 : u + g + f
            sleep(0.005) # t2 = 0.05
            if not ret.is_success():
                print(f"move_servo_t 실패 ", ret)
                
        except Exception as e:
            print(f"T-servo Failed ..! {e}")

        viewer.sync()






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