from time import time, sleep
from copy import deepcopy
import os 
import sys

import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

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

class RB3_tservo_ROS(Node):
    def __init__(s):
        super().__init__('rb3_torque_servoing')
        #----- Real Robot Setup -----
        ROBOT_ADDRESS = "192.168.1.200"

        
        try:
            s.robot= rb.Cobot(ROBOT_ADDRESS)
            
            s.rc = rb.ResponseCollector()
            
            s.robot_data = CobotData(ROBOT_ADDRESS)

            s.state = s.robot_data.request_data()
    
            s.robot.set_operation_mode(s.rc, rb.OperationMode.Real)
         
            s.robot.set_speed_bar(s.rc, 0.3)
        except:
            print("No Robot Connetion!")

        #s.robot.set_freedrive_mode(rc, on=False)
        s.t1 = 0.01 #이동시간
        s.t2 = 0.05 #유지시간
            
        #----- Model Define -----
        model_path="/home/kdh/Desktop/delto/delto_description2/rb3_single/scene_rb3.xml"
        s.m = mujoco.MjModel.from_xml_path(model_path)
        s.d = mujoco.MjData(s.m)

        s.M = np.zeros((s.m.nv, s.m.nv), dtype=np.float64)
        s.G = np.zeros((s.m.nv), dtype=np.float64)
        
        s.viewer = mujoco.viewer.launch_passive(s.m, s.d) # viewer initialize
        
        #----- Jacobian Compute Variable -----
        s._jacp = np.zeros((3, s.m.nv), dtype=np.float64) # compute
        s._jacr = np.zeros((3, s.m.nv), dtype=np.float64) 
        
        s.jacp= [np.zeros((3, s.m.nv), dtype=np.float64) for _ in range(8)] # save
        s.jacr = [np.zeros((3, s.m.nv), dtype=np.float64) for _ in range(8)] 
        
        #----- Site ID -----
        s.tcp_site_id = s.m.site("tcp").id

        
        #----- VSD Gain -----
        s.K = 10.0 # Gripper Posiotion Control Gain
        s.zeta = 0.1
           
        s.K_p = 90.0 # Position Control Gain 
        s.zeta_p = 2.0
           
        s.K_o = 5.0 # Orientation Control Gain 
        s.zeta_o = 1.0
        
        #----- Dynamics Compute Variable -----
        s.xpos_err = [np.zeros(3) for _ in range(8)]
        s.xpos_dot = [np.zeros(3) for _ in range(8)]
        s.force = [np.zeros(3) for _ in range(8)]
        
        s.ori_err = []
        
        s.w0 = []
        
        s.F_ori = []
        
        s.prev_jpos = None

        #----- Control Input Torque -----
        s.torque0 = np.zeros(6)
        
        #----- Joint Damping Coefficient -----
        s.C0 = np.zeros((6,6)) 

        #----- Joint Friction Coefficient -----
        s.Cf_RB = np.array([10.5, 6.5, 6.5, 0.7, 0.7, 0.9]) # coulomb friction coef
        s.Vf_RB = np.array([10.5, 6.5, 6.5, 0.7, 0.7, 0.9]) # viscous friction coef
        
        s.fric_coef_RB = 8*1e-1 # friction curve coef
     
        s.Tf = np.zeros((6))
        
        #----- Desired Pos/Rot Value -----
        #TODO <- ROS Call back 
        #----- Goal        
        s.target_pose = PoseStamped()

        
        s.pose_received = False
        
        s.tcp_goal = s.create_subscription(PoseStamped, '/tcp_goal', s.tcp_goal_register, 10)
       
        s.desired_xpos_tcp = np.array([0.0, -0.3, 0.1]) #TODO
        # s.desired_xpos_ee = np.array([0.4, -0.2, 0.0]) #TODO

        s.desired_rpy = np.array([90.0, 0.0, 0.0]) #TODO # roll, pitch, yaw 입력
        

        #----- Control Loop Call ------
        if hasattr(RB3_tservo_ROS,"robot"):
            s.initial_robot_pose()
            
        loop_hz = 1000  #hz
        s.Control_Loop = s.create_timer((1/loop_hz), s.control_loop)

        s.now = None # loop dt calc val
        s.prev_time = None

        s.hz_window = []
    
        # ------- Small Error Compensator Flag TEST-----
        s.r_reach = False
        s.l_reach = False
        s.new_target = False
        
    #----- Callback Func -----
    def tcp_goal_register(s, msg):
        print("tcp goal input")
        s.K_p = 90
        if msg.header.frame_id == 'tcp_goal':
            s.target_pose = msg
            s.target_pose.header.frame_id = 'world'
            # s.r_pose_received = True
            s.get_logger().info("target pose input.")
            
            s.desired_xpos_tcp[0] = s.target_pose.pose.position.x
            s.desired_xpos_tcp[1] = s.target_pose.pose.position.y
            s.desired_xpos_tcp[2] = s.target_pose.pose.position.z
            
            qx = s.target_pose.pose.orientation.x
            qy = s.target_pose.pose.orientation.y
            qz = s.target_pose.pose.orientation.z
            qw = s.target_pose.pose.orientation.w
            q = np.array([qx, qy, qz, qw])
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

            rpy = R.from_quat(q).as_euler('xyz', degrees=True)  # roll, pitch, yaw
            roll, pitch, yaw = rpy
            s.desired_rpy = np.array([roll, pitch, yaw], dtype=float)

            print(f"New TCP Goal Registered {s.desired_xpos_tcp} {s.desired_rpy}")
        
    #----- Compute Func ------    
    def compute_jacobian(s):
        mujoco.mj_forward(s.m, s.d)        
        
        mujoco.mj_jacSite(s.m, s.d, s._jacp, s._jacr, s.tcp_site_id)
        s.jacp[0] = deepcopy(s._jacp[:, 0:6])
        s.jacr[0] = deepcopy(s._jacr[:, 0:6])

    def compute_joint_damping(s):
        # mujoco.mj_step(s.m, s.d)
        # mujoco.mj_fullM(s.m, s.M, s.d.qM)
        # mujoco.mj_forward(s.m, s.d)
        np.fill_diagonal(s.C0, np.sum(np.abs(s.M[0:6, 0:6]), axis=1))
 
    
    def compute_position_error(s):
        s.xpos_err[0] = s.d.site("tcp").xpos - s.desired_xpos_tcp
   
    
    def compute_linear_velocity(s):
        s.xpos_dot[0] = s.jacp[0] @ s.d.qvel[0:6]
    
    def compute_position_control_force(s):
        a, b, c, e, f = [6.0, 2.1, 2.5, 1, 0.2]#[7.2, 2.1, 2.3, 1, 0.05] #a=7.5    
        Ecurv = (a / (b*np.exp(c * np.abs(s.xpos_err[0]) + e)))+f
        
        s.force[0] = (s.K_p * s.xpos_err[0]) + (s.zeta_p * np.sqrt(s.K_p) * s.xpos_dot[0])
        s.force[0] = Ecurv * s.force[0]
        
        
    def compute_orientation_error(s):
        R_current = s.d.site("tcp").xmat.reshape(3, 3)

        R_desired = rpy_to_rotmat(*s.desired_rpy)


        s.ori_err = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        ) # e  = (x X xd) + (y X yd) + (z X zd)

    def compute_angular_velocity(s):
        # Angular Velocity
        s.w0 = s.jacr[0] @ s.d.qvel[0:6]
        # print(f"wo : {w0}")
    
    def compute_orientation_control_force(s):
        s.F_ori = (s.K_o * s.ori_err) #+ (zeta_o * np.sqrt(K_o) * w0)
    
    def compute_gravity_compensation(s):
        # mujoco.mj_step(s.m, s.d)
        # mujoco.mj_fullM(s.m, s.M, s.d.qM)

        qvel_backup = deepcopy(s.d.qvel)
        ###
        s.d.qvel[:] = 0
        mujoco.mj_forward(s.m, s.d)
        mujoco.mj_rne(s.m, s.d, 0, s.G) 
        s.d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(s.m, s.d)
        
    def compute_friction_compensation(s):
        mujoco.mj_forward(s.m, s.d)

        for i in range(6): # RB
            s.Tf[i] = (s.Cf_RB[i] * np.tanh(s.fric_coef_RB*s.d.qvel[i]) + s.Vf_RB[i] * s.d.qvel[i])        
    
    def compute_control_torque(s):
        max_torque = 40
        
        s.torque0 = np.clip((- 0 * s.C0 @ s.d.qvel[0:6] 
                             - 1 * s.jacp[0].T @ s.force[0] 
                             - 0 * s.jacr[0].T @ s.F_ori
                             + 1 * s.G[0:6] 
                             + 0 * s.Tf[0:6])
                    ,-max_torque, max_torque)
        
    #----- vis -----
    def visualize_sync(s):
        s.d.ctrl[0:6] = s.torque0
        
        mujoco.mj_forward(s.m, s.d)        
        
    #----- Robot I/O -----
    def robot_command(s):
        try:
            ret = s.robot.move_servo_t(s.rc, s.torque0, s.t1, s.t2, compensation=2)
            # # comp 0 : u / 1 : u + g / 2 : u + f / 3 : u + g + f
            sleep(0.005) # t2 = 0.05
            if not ret.is_success():
                # print(f"move_servo_t 실패 ", ret_L, ret_R)
                print(f"move_servo_t 실패 ")
                
        except Exception as e:
            print(f"T-servo Failed ..! {e}")
    

    def initial_robot_pose(s):
        try:
            jpos, jvel = s.rb_get_joint_state()
            print(f"Initial jpos : {jpos} | jvel : {jvel}")

            while (len(jpos)==0):
                print("Waiting for robot init data..")
                s.d.qpos[0:6] = np.deg2rad(jpos)
                s.d.qvel[:] = 0 #TODO

        except Exception as e:
            print(f"Cannot get Robot init data ..! {e}")
            s.d.qpos[0:6] = [0.5, 0.3, -1.3, 0.4, -1.57, 0.0]
            s.d.qvel[:] = 0

    def rb_get_joint_state(s):
        jpos = []
        jvel = []

        for i in range(6):
            var_pos = getattr(SystemVariable, f"SD_J{i}_ANG")
            var_vel = getattr(SystemVariable, f"SD_J{i}_VEL")
            _, pos = s.robot.get_system_variable(s.rc, var_pos)
            _, vel = s.robot.get_system_variable(s.rc, var_vel)

            jpos.append(pos)
            jvel.append(vel)

        return np.array(jpos), np.array(jvel)
        
    def rb_get_joint_position(s):
        jpos = []
        if s.state is None:
            print("Failed to get robot state.")
        else:
            jpos = s.state.sdata.jnt_ang  # 조인트 위치 (deg)
            # print(f"R:{jpos_R} | L:{jpos_L}")
        return np.array(jpos)
    
    def robot_data_update(s,loop_dt):
        try:
            ## real data update ##
            jpos = s.rb_get_joint_position()
            
            jvel = np.zeros(6)
            if s.prev_jpos is not None:
                jvel = (jpos - s.prev_jpos) / loop_dt
                # print(f"jvel calc :: \nR : {jvel_R}\nL : {jvel_L}")
            s.prev_jpos = jpos
            
            # print(f"JP : {jpos} | JV : {jvel}")
            s.d.qpos[0:6] = np.deg2rad(jpos)
            s.d.qvel[0:6] = np.deg2rad(jvel)
            mujoco.mj_forward(s.m, s.d)  
        except Exception as e:
            print(f"real data update failed..! {e}")
    
    def get_robot_data(s):
         # while viewer.is_running(): #rclpy 랑 겹치...?
        try:
            state = s.robot_data.request_data() #매 루프 데이터 업데이트
            
            if state.sdata.op_stat_collision_occur:
                print("Robot in Collision")
                rclpy.shutdown() #break
            if state.sdata.op_stat_sos_flag==4:
                print(f"Command Input Error | JVEL: {s.d.qvel[0:6]}")
                rclpy.shutdown() #break
        except:
            pass
      
    #----- Main Control ----
    def control_loop(s):
        if not s.viewer.is_running():
            s.get_logger().info("Viewer closed. Stopping control loop.")
            s.Control_Loop.cancel() 
            os._exit(0)
            sys.exit(0)
            return # Viewer 동기화 탈출
        
        try:
            s.state = s.robot_data.request_data() #매 루프 데이터 업데이트
        
            if s.state.sdata.op_stat_collision_occur:
                print("Robot in Collision")
                rclpy.shutdown() #break
                os._exit(0)
            if s.state.sdata.op_stat_sos_flag==4:
                # print(f"Command Input Error | JVEL R: {jvel_L} JVEL L: {jvel_L}")
                rclpy.shutdown() #break
                os._exit(0)
        except:
            pass
        
        s.now = time()
        if s.prev_time is None:
            s.prev_time = s.now
            return    
        loop_dt = s.now - s.prev_time
        s.prev_time = s.now

        mujoco.mj_step(s.m, s.d)
        mujoco.mj_fullM(s.m, s.M, s.d.qM)
        
        try:
        ## real data update ##
            jpos = s.rb_get_joint_position()
            jvel = np.zeros(6)
            if s.prev_jpos is not None:
                jvel = (jpos - s.prev_jpos) / loop_dt
                # print(f"jvel calc :: \nR : {jvel_R}\nL : {jvel_L}")
            s.prev_jpos = jpos
            
            # print(f"JP : {jpos} | JV : {jvel}")
            s.d.qpos[0:6] = np.deg2rad(jpos)
            s.d.qvel[0:6] = np.deg2rad(jvel)
            mujoco.mj_forward(s.m, s.d)  
        except Exception as e:
            pass
            # print(f"real data update failed..! {e}")
        
        #----------------------------------------
        s.compute_gravity_compensation()
        
        s.compute_joint_damping()
                
        s.compute_jacobian()        
        
        s.compute_position_error()
        s.compute_linear_velocity()
        s.compute_position_control_force()
        
        s.compute_orientation_error()
        s.compute_angular_velocity()
        s.compute_orientation_control_force()
        

        s.compute_friction_compensation()
        
        #TODO s.compute_virtual_repulsion()
        
        s.compute_control_torque()
        #----------------------------------------
        # TODO RB Postion control mode         
        #----------------------------------------
        s.visualize_sync()
        
        s.robot_command()
        
                
        s.viewer.sync()
        
        #----------------------------------------
        # if hasattr(s, "state_R") and hasattr(s, "state_L"): # state 있을때만 실행 체크
        # if newgoaled and exec status false.. -> status true -> newgoaled fals?
        
        
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            s.hz_window.append(hz)
            if len(s.hz_window) > 30:  
                s.hz_window.pop(0)
            # print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(s.hz_window):.2f})")
        
def main(args=None):
    rclpy.init(args=args)
    node = RB3_tservo_ROS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()