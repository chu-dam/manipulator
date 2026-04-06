from time import time, sleep
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np


STATE_PREINSERT = 0
STATE_ENTRY = 1
STATE_DONE = 2


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
    return Rz @ Ry @ Rx


def create_model_and_data(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d


def initialize_robot_state(d):
    d.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    d.qpos[6:] = [0.010, 0.010]   # 초기 gripper open


def create_work_buffers(m):
    M = np.zeros((m.nv, m.nv), dtype=np.float64)
    G = np.zeros((m.nv), dtype=np.float64)

    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)

    C0 = np.zeros((6, 6))
    return M, G, jacp, jacr, C0


def compute_mass_and_gravity(m, d, M, G):
    mujoco.mj_fullM(m, M, d.qM)

    qvel_backup = deepcopy(d.qvel)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    mujoco.mj_rne(m, d, 0, G)
    d.qvel[:] = qvel_backup[:]
    mujoco.mj_forward(m, d)


def compute_task_torque(
    m, d,
    M, G, jacp, jacr, C0,
    desired_xpos_tcp, desired_rpy,
    K_a, zeta_a, K_o, zeta_o
):
    np.fill_diagonal(C0, np.sum(np.abs(M[0:6, 0:6]), axis=1))

    tcp_site_id = m.site("tcp").id
    mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

    jacp0 = deepcopy(jacp[:, 0:6])
    jacr0 = deepcopy(jacr[:, 0:6])

    # Position Error
    xpos_err0 = d.site("tcp").xpos - desired_xpos_tcp

    # Orientation Error
    R_current = d.site(tcp_site_id).xmat.reshape(3, 3)
    R_desired = rpy_to_rotmat(*desired_rpy)

    ori_err0 = (
        np.cross(R_desired[:, 0], R_current[:, 0]) +
        np.cross(R_desired[:, 1], R_current[:, 1]) +
        np.cross(R_desired[:, 2], R_current[:, 2])
    )

    # Angular Velocity
    w0 = jacr0 @ d.qvel[0:6]

    # Orientation Force
    F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

    # Linear Velocity
    xpos_dot0 = jacp0 @ d.qvel[0:6]

    # Linear Force
    force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

    M0 = M[0:6, 0:6]
    M0_inv = np.linalg.pinv(M0)

    eps = 1e-6
    Lambda_inv = jacp0 @ M0_inv @ jacp0.T
    Lambda_pos = np.linalg.pinv(Lambda_inv + eps * np.eye(3))

    force0_task = Lambda_pos @ force0

    # Torque (원본 그대로 유지)
    torque0 = (
        -0 * C0 @ d.qvel[0:6]
        -1 * jacp0.T @ force0_task
        +1 * G[0:6]
        -1 * jacr0.T @ F_ori_0
    )

    return torque0


def apply_control(d, torque0, max_torque=80, gripper_cmd=0.012):
    d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
    d.ctrl[6] = gripper_cmd   # open
    # d.ctrl[6] = 0.000       # close


def get_site_xpos_or_fallback(d, site_name, fallback):
    try:
        return d.site(site_name).xpos.copy()
    except KeyError:
        return np.array(fallback, dtype=np.float64)


def get_target_by_state(d, state):
    """
    상태에 따라 목표 위치만 바뀌고,
    자세와 그리퍼 명령은 지금은 고정
    hole 관련 site가 없으면 fallback 좌표 사용
    """

    # fallback 좌표는 현재 작업 환경에 맞게 나중에 조정
    hole_preinsert = get_site_xpos_or_fallback(
        d, "hole_preinsert", [0.0, -0.62, 0.32]
    )
    hole_entry = get_site_xpos_or_fallback(
        d, "hole_entry", [0.0, -0.68, 0.10]
    )

    desired_rpy = np.array([90.0, 0.0, 0.0])
    gripper_cmd = 0.012

    if state == STATE_PREINSERT:
        desired_xpos_tcp = hole_preinsert
    elif state == STATE_ENTRY:
        desired_xpos_tcp = hole_entry
    else:
        desired_xpos_tcp = hole_entry

    return desired_xpos_tcp, desired_rpy, gripper_cmd


def update_state(d, state, pos_tol=0.01):
    tcp_pos = d.site("tcp").xpos.copy()

    hole_preinsert = get_site_xpos_or_fallback(
        d, "hole_preinsert", [0.0, -0.62, 1.02]
    )
    hole_entry = get_site_xpos_or_fallback(
        d, "hole_entry", [0.0, -0.68, 1.02]
    )

    if state == STATE_PREINSERT:
        err = np.linalg.norm(tcp_pos - hole_preinsert)
        if err < pos_tol:
            print("[STATE] PREINSERT -> ENTRY")
            return STATE_ENTRY

    elif state == STATE_ENTRY:
        err = np.linalg.norm(tcp_pos - hole_entry)
        if err < pos_tol:
            print("[STATE] ENTRY -> DONE")
            return STATE_DONE

    return state


def get_state_name(state):
    if state == STATE_PREINSERT:
        return "PREINSERT"
    elif state == STATE_ENTRY:
        return "ENTRY"
    else:
        return "DONE"


def print_debug(d, state, desired_xpos_tcp, print_orientation=False):
    current_tcp_pos = d.site("tcp").xpos
    pos_err = current_tcp_pos - desired_xpos_tcp

    msg = (
        f"[{get_state_name(state)}] "
        f"TCP : X={current_tcp_pos[0]:.4f}, "
        f"Y={current_tcp_pos[1]:.4f}, "
        f"Z={current_tcp_pos[2]:.4f} | "
        f"ERR : dX={pos_err[0]:.4f}, "
        f"dY={pos_err[1]:.4f}, "
        f"dZ={pos_err[2]:.4f}"
    )
    print(msg)


def main():
    model_path = "/home/chu/manipulator_control/rb5/scene_rb5_gripper.xml"
    m, d = create_model_and_data(model_path)

    initialize_robot_state(d)

    M, G, jacp, jacr, C0 = create_work_buffers(m)

    K_a = np.array([50.0, 50.0, 35.0])
    zeta_a = np.array([9.0, 9.0, 4.0])

    K_o = np.array([3.0, 3.0, 1.5])
    zeta_o = np.array([0.3, 0.3, 0.2])

    d.ctrl[6] = 0.012   # open

    state = STATE_PREINSERT

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time()
        while viewer.is_running():
            t = time() - t0

            desired_xpos_tcp, desired_rpy, gripper_cmd = get_target_by_state(d, state)

            compute_mass_and_gravity(m, d, M, G)

            torque0 = compute_task_torque(
                m, d,
                M, G, jacp, jacr, C0,
                desired_xpos_tcp, desired_rpy,
                K_a, zeta_a, K_o, zeta_o
            )

            apply_control(d, torque0, max_torque=80, gripper_cmd=gripper_cmd)

            print_debug(d, state, desired_xpos_tcp)

            state = update_state(d, state, pos_tol=0.01)

            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == "__main__":
    main()