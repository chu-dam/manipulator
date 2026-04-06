from time import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np


STATE_APPROACH_ABOVE_PEG = 0
STATE_APPROACH_DOWN_TO_PEG = 1
STATE_GRASP = 2
STATE_LIFT = 3
STATE_MOVE_PREINSERT = 4
STATE_ENTRY = 5
STATE_INSERT = 6
STATE_RELEASE = 7
STATE_RETREAT = 8
STATE_DONE = 9


def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
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


def validate_required_sites(m):
    required_sites = [
        "tcp",
        "grasp_center",
        "peg_center",
        "peg_tip",
    ]

    missing = []
    for name in required_sites:
        try:
            _ = m.site(name).id
        except KeyError:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"필수 site가 없습니다: {missing}\n"
            f"rb5_gripper.xml / scene_rb5_gripper.xml의 site 이름을 확인하세요."
        )


def initialize_robot_state(m, d):
    # arm home
    d.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]

    # gripper joint qpos address
    left_id = m.joint("grip_left").id
    right_id = m.joint("grip_right").id

    left_qpos_adr = m.jnt_qposadr[left_id]
    right_qpos_adr = m.jnt_qposadr[right_id]

    d.qpos[left_qpos_adr] = 0.010
    d.qpos[right_qpos_adr] = 0.010

    d.qvel[:] = 0.0


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
    desired_xpos_site, desired_rpy,
    K_a, zeta_a, K_o, zeta_o,
    site_name="tcp"
):
    np.fill_diagonal(C0, np.sum(np.abs(M[0:6, 0:6]), axis=1))

    site_id = m.site(site_name).id
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)

    jacp0 = deepcopy(jacp[:, 0:6])
    jacr0 = deepcopy(jacr[:, 0:6])

    xpos_err0 = d.site(site_name).xpos - desired_xpos_site

    R_current = d.site(site_id).xmat.reshape(3, 3)
    R_desired = rpy_to_rotmat(*desired_rpy)

    ori_err0 = (
        np.cross(R_desired[:, 0], R_current[:, 0]) +
        np.cross(R_desired[:, 1], R_current[:, 1]) +
        np.cross(R_desired[:, 2], R_current[:, 2])
    )

    w0 = jacr0 @ d.qvel[0:6]
    F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

    xpos_dot0 = jacp0 @ d.qvel[0:6]
    force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

    M0 = M[0:6, 0:6]
    M0_inv = np.linalg.pinv(M0)

    eps = 1e-6
    Lambda_inv = jacp0 @ M0_inv @ jacp0.T
    Lambda_pos = np.linalg.pinv(Lambda_inv + eps * np.eye(3))

    force0_task = Lambda_pos @ force0

    torque0 = (
        -0 * C0 @ d.qvel[0:6]
        -1 * jacp0.T @ force0_task
        +1 * G[0:6]
        -1 * jacr0.T @ F_ori_0
    )

    return torque0


def apply_control(d, torque0, max_torque=80, gripper_cmd=0.012):
    d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
    d.ctrl[6] = gripper_cmd


def get_state_name(state):
    if state == STATE_APPROACH_ABOVE_PEG:
        return "APPROACH_ABOVE_PEG"
    elif state == STATE_APPROACH_DOWN_TO_PEG:
        return "APPROACH_DOWN_TO_PEG"
    elif state == STATE_GRASP:
        return "GRASP"
    elif state == STATE_LIFT:
        return "LIFT"
    else:
        return "DONE"




def get_target_by_state(d, state, state_data):
    desired_rpy = np.array([90.0, 0.0, 0.0])

    peg_center = d.site("peg_center").xpos.copy()
    peg_above = peg_center + np.array([0.0, 0.0, 0.05])
    peg_pregrasp = peg_center + np.array([0.0, 0.0, 0.013])   # 월드 z + 13 mm

    if state == STATE_APPROACH_ABOVE_PEG:
        desired_xpos_site = peg_above
        control_site_name = "grasp_center"
        gripper_cmd = 0.012   # open

    elif state == STATE_APPROACH_DOWN_TO_PEG:
        desired_xpos_site = peg_pregrasp
        control_site_name = "grasp_center"
        gripper_cmd = 0.012   # still open

    elif state == STATE_GRASP:
        desired_xpos_site = peg_pregrasp
        control_site_name = "grasp_center"
        gripper_cmd = 0.000   # close

    elif state == STATE_LIFT:
        desired_xpos_site = state_data["lift_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    else:
        desired_xpos_site = state_data["lift_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    return desired_xpos_site, desired_rpy, gripper_cmd, control_site_name


def get_peg_freejoint_addrs(m):
    body_id = m.body("peg").id
    jnt_adr = m.body_jntadr[body_id]

    if jnt_adr < 0:
        raise RuntimeError("peg body에 joint가 없습니다.")
    if m.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("peg는 freejoint body여야 합니다.")

    qpos_adr = m.jnt_qposadr[jnt_adr]
    qvel_adr = m.jnt_dofadr[jnt_adr]
    return qpos_adr, qvel_adr


def attach_peg_to_gripper(m, d, state_data):
    """
    semi-physical grasp:
    grasp 성공 순간의 상대 위치를 유지하면서 peg를 gripper에 붙여서 따라오게 함
    """
    qpos_adr = state_data["peg_qpos_adr"]
    qvel_adr = state_data["peg_qvel_adr"]

    grasp_center_pos = d.site("grasp_center").xpos.copy()

    # grasp 성공 순간 저장한 상대 위치를 그대로 유지
    peg_origin_pos = grasp_center_pos + state_data["peg_body_offset_from_grasp_world"]

    d.qpos[qpos_adr:qpos_adr+3] = peg_origin_pos
    d.qpos[qpos_adr+3:qpos_adr+7] = state_data["peg_quat_locked"]

    d.qvel[qvel_adr:qvel_adr+6] = 0.0


def set_state(new_state, state_data, d, now):
    state_data["state"] = new_state
    state_data["state_enter_time"] = now

    if new_state == STATE_LIFT:
        current_grasp_center = d.site("grasp_center").xpos.copy()
        state_data["lift_target"] = current_grasp_center + np.array([0.0, 0.0, 0.08])


def maybe_lock_peg(m, d, state_data):
    if state_data["grasped"]:
        return

    grasp_center = d.site("grasp_center").xpos.copy()
    peg_center = d.site("peg_center").xpos.copy()

    dx = grasp_center[0] - peg_center[0]
    dy = grasp_center[1] - peg_center[1]
    time_in_state = state_data["time_now"] - state_data["state_enter_time"]

    # x, y 오차만으로 grasp 판정
    if abs(dx) < 0.004 and abs(dy) < 0.004 and time_in_state > 0.35:
        state_data["grasped"] = True

        qpos_adr = state_data["peg_qpos_adr"]

        # 현재 peg quaternion 고정 저장
        peg_quat_locked = d.qpos[qpos_adr+3:qpos_adr+7].copy()
        state_data["peg_quat_locked"] = peg_quat_locked

        # 핵심:
        # grasp 성공 순간의 "peg body origin - grasp_center" 상대 위치를 그대로 저장
        peg_body_pos = d.body("peg").xpos.copy()
        peg_body_offset_from_grasp_world = peg_body_pos - grasp_center
        state_data["peg_body_offset_from_grasp_world"] = peg_body_offset_from_grasp_world

        print(f"[GRASP] peg locked to gripper | dx={dx:.6f}, dy={dy:.6f}")


def update_state(m, d, state_data):
    state = state_data["state"]
    now = state_data["time_now"]

    grasp_center = d.site("grasp_center").xpos.copy()
    peg_center = d.site("peg_center").xpos.copy()
    peg_above = peg_center + np.array([0.0, 0.0, 0.05])
    peg_pregrasp = peg_center + np.array([0.0, 0.0, 0.013])


    if state == STATE_APPROACH_ABOVE_PEG:
        dx = grasp_center[0] - peg_above[0]
        dy = grasp_center[1] - peg_above[1]
        dz = grasp_center[2] - peg_above[2]

        if abs(dx) <= 0.0001 and abs(dy) <= 0.0001 and abs(dz) <= 0.0001:
            print("[STATE] APPROACH_ABOVE_PEG -> APPROACH_DOWN_TO_PEG")
            set_state(STATE_APPROACH_DOWN_TO_PEG, state_data, d, now)

    elif state == STATE_APPROACH_DOWN_TO_PEG:
        err = np.linalg.norm(grasp_center - peg_pregrasp)
        if err < 0.004:
            print("[STATE] APPROACH_DOWN_TO_PEG -> GRASP")
            set_state(STATE_GRASP, state_data, d, now)

    elif state == STATE_GRASP:
        maybe_lock_peg(m, d, state_data)

        if state_data["grasped"]:
            print("[STATE] GRASP -> LIFT")
            set_state(STATE_LIFT, state_data, d, now)

    elif state == STATE_LIFT:
        current = d.site("grasp_center").xpos.copy()
        target = state_data["lift_target"]
        err = np.linalg.norm(current - target)

        if err < 0.005:
            print("[STATE] LIFT -> DONE")
            set_state(STATE_DONE, state_data, d, now)


def print_debug(d, state_data, desired_xpos_site, control_site_name, print_every=0.1):
    now = state_data["time_now"]
    if now - state_data["last_print_time"] < print_every:
        return
    state_data["last_print_time"] = now

    current_pos = d.site(control_site_name).xpos.copy()
    peg_center = d.site("peg_center").xpos.copy()
    peg_tip = d.site("peg_tip").xpos.copy()
    pos_err = current_pos - desired_xpos_site

    print(
        f"[{get_state_name(state_data['state'])} | site={control_site_name}] "
        f"POS=({current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}) | "
        f"ERR=({pos_err[0]:.4f}, {pos_err[1]:.4f}, {pos_err[2]:.4f}) | "
        f"PEG_CENTER=({peg_center[0]:.4f}, {peg_center[1]:.4f}, {peg_center[2]:.4f}) | "
        f"PEG_TIP=({peg_tip[0]:.4f}, {peg_tip[1]:.4f}, {peg_tip[2]:.4f}) | "
        f"grasped={state_data['grasped']}"
    )


def main():
    model_path = "/home/chu/manipulator_control/rb5/scene_rb5_gripper.xml"
    m, d = create_model_and_data(model_path)

    validate_required_sites(m)
    initialize_robot_state(m, d)

    M, G, jacp, jacr, C0 = create_work_buffers(m)

    peg_qpos_adr, peg_qvel_adr = get_peg_freejoint_addrs(m)

    K_a = np.array([50.0, 50.0, 35.0])
    zeta_a = np.array([9.0, 9.0, 4.0])

    K_o = np.array([3.0, 3.0, 1.5])
    zeta_o = np.array([0.3, 0.3, 0.2])

    state_data = {
        "state": STATE_APPROACH_ABOVE_PEG,
        "state_enter_time": 0.0,
        "time_now": 0.0,
        "last_print_time": -1.0,
        "grasped": False,
        "lift_target": np.zeros(3),
        "peg_qpos_adr": peg_qpos_adr,
        "peg_qvel_adr": peg_qvel_adr,
        "peg_quat_locked": np.array([1.0, 0.0, 0.0, 0.0]),
        "peg_body_offset_from_grasp_world": np.zeros(3),
    }

    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time()

        while viewer.is_running():
            now = time() - t0
            state_data["time_now"] = now

            desired_xpos_site, desired_rpy, gripper_cmd, control_site_name = get_target_by_state(
                d, state_data["state"], state_data
            )

            compute_mass_and_gravity(m, d, M, G)

            torque0 = compute_task_torque(
                m, d,
                M, G, jacp, jacr, C0,
                desired_xpos_site, desired_rpy,
                K_a, zeta_a, K_o, zeta_o,
                site_name=control_site_name
            )

            apply_control(d, torque0, max_torque=80, gripper_cmd=gripper_cmd)

            # grasp된 이후에는 peg를 gripper에 붙여서 따라오게 함
            if state_data["grasped"]:
                attach_peg_to_gripper(m, d, state_data)
                mujoco.mj_forward(m, d)

            print_debug(d, state_data, desired_xpos_site, control_site_name, print_every=0.1)

            update_state(m, d, state_data)

            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == "__main__":
    main()