import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb
import numpy as np

######## torque servo
try:
    ROBOT_ADDRESS = "192.168.1.200"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()

    robot_data = CobotData(ROBOT_ADDRESS)
    
    state = robot_data.request_data()
    # robot.set_freedrive_mode(True)
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.3)
    robot.set_freedrive_mode(rc, on=True)
    t1 = 0.01 #이동시간
    t2 = 0.05 #유지시간
    
except Exception as e:
    print(f"No Robot Connection ..! {e}")

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
    time.sleep(3)
except Exception as e:
    print(f"Cannot get Robot init data ..! {e}")
    d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
    d.qvel[:] = 0
    time.sleep(3)


M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)


######################
prev_time = time.time()
hz_window = []
prev_jpos = None  

with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time.time()
    while viewer.is_running():
        state = robot_data.request_data() #매 루프 데이터 업데이트
        
        # move_servo_t 호출 전 시간 기록
        now = time.time()
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
                print(f"jvel calc :: {jvel} = {jpos}-{prev_jpos}/{loop_dt}")
            prev_jpos = jpos
            
            # print(f"JP : {jpos} | JV : {jvel}")
            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)
            mujoco.mj_forward(m, d)    
        except Exception as e:
            print(f"real data update failed..! {e}")

         # Hz 측정 (1 / 주기)
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)
            if len(hz_window) > 30:  # 최근 30프레임 평균
                hz_window.pop(0)
            # print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f}) |\nTaget torque : {d.ctrl[0:6]}")
       
        time.sleep(0.01)  
        viewer.sync() 

