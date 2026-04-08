from time import time
from copy import deepcopy
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


STATE_APPROACH_ABOVE_PEG = 0
STATE_APPROACH_DOWN_TO_PEG = 1
STATE_GRASP = 2
STATE_LIFT = 3
STATE_MOVE_PREINSERT = 4
STATE_CONTACT_APPROACH = 5
STATE_HOLE_SEARCH = 6
STATE_WIGGLE = 7
STATE_SCREW = 8
STATE_INSERT_FINAL = 9
STATE_RELEASE = 10
STATE_RETREAT = 11
STATE_DONE = 12


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
        "hole_preinsert",
        "hole_entry",
        "hole_bottom",
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


def validate_required_geoms(m):
    required_geoms = [
        "left_finger_geom",
        "right_finger_geom",
        "peg_geom",
    ]

    missing = []
    for name in required_geoms:
        try:
            _ = m.geom(name).id
        except KeyError:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"필수 geom이 없습니다: {missing}\n"
            f"rb5_gripper.xml의 geom 이름을 확인하세요."
        )


def initialize_robot_state(m, d):
    # arm home
    d.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]

    # gripper open
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
    site_name="grasp_center"
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


def apply_control(m, d, torque0, state_data, max_torque=80, gripper_cmd_target=0.012):
    d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)

    current = state_data["gripper_cmd_current"]

    if gripper_cmd_target < current:
        speed = state_data["gripper_close_speed"]
    else:
        speed = state_data["gripper_open_speed"]

    max_delta = speed * m.opt.timestep
    delta = np.clip(gripper_cmd_target - current, -max_delta, max_delta)

    current = current + delta
    state_data["gripper_cmd_current"] = current

    d.ctrl[state_data["gripper_act_id"]] = current


def get_state_name(state):
    if state == STATE_APPROACH_ABOVE_PEG:
        return "APPROACH_ABOVE_PEG"
    elif state == STATE_APPROACH_DOWN_TO_PEG:
        return "APPROACH_DOWN_TO_PEG"
    elif state == STATE_GRASP:
        return "GRASP"
    elif state == STATE_LIFT:
        return "LIFT"
    elif state == STATE_MOVE_PREINSERT:
        return "MOVE_PREINSERT"
    elif state == STATE_CONTACT_APPROACH:
        return "CONTACT_APPROACH"
    elif state == STATE_HOLE_SEARCH:
        return "HOLE_SEARCH"
    elif state == STATE_WIGGLE:
        return "WIGGLE"
    elif state == STATE_SCREW:
        return "SCREW"
    elif state == STATE_INSERT_FINAL:
        return "INSERT_FINAL"
    elif state == STATE_RELEASE:
        return "RELEASE"
    elif state == STATE_RETREAT:
        return "RETREAT"
    else:
        return "DONE"


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


def get_contact_pair_force(m, d, geom_a_id, geom_b_id):
    wrench = np.zeros(6, dtype=np.float64)

    contact_count = 0
    normal_force_sum = 0.0
    total_force_sum = 0.0

    for i in range(d.ncon):
        con = d.contact[i]

        if hasattr(con, "geom"):
            g1 = int(con.geom[0])
            g2 = int(con.geom[1])
        else:
            g1 = int(con.geom1)
            g2 = int(con.geom2)

        pair_match = (
            (g1 == geom_a_id and g2 == geom_b_id) or
            (g1 == geom_b_id and g2 == geom_a_id)
        )
        if not pair_match:
            continue

        if con.exclude != 0 or con.efc_address < 0:
            continue

        mujoco.mj_contactForce(m, d, i, wrench)

        normal_force = abs(wrench[0])
        total_force = np.linalg.norm(wrench[:3])

        contact_count += 1
        normal_force_sum += normal_force
        total_force_sum += total_force

    return contact_count, normal_force_sum, total_force_sum


def attach_peg_to_gripper(m, d, state_data):
    """
    sim 편의용:
    grasp 성공 후에는 peg를 gripper에 붙여서 따라오게 함.
    제어 로직은 현재 peg 좌표를 다시 읽지 않음.
    """
    qpos_adr = state_data["peg_qpos_adr"]
    qvel_adr = state_data["peg_qvel_adr"]

    grasp_center_pos = d.site("grasp_center").xpos.copy()
    peg_origin_pos = grasp_center_pos + state_data["peg_body_offset_from_grasp_nominal_world"]

    d.qpos[qpos_adr:qpos_adr+3] = peg_origin_pos
    d.qpos[qpos_adr+3:qpos_adr+7] = state_data["peg_init_quat_world"]
    d.qvel[qvel_adr:qvel_adr+6] = 0.0


def set_state(new_state, state_data, d, now):
    state_data["state"] = new_state
    state_data["state_enter_time"] = now
    state_data["phase_start_pos"] = d.site("grasp_center").xpos.copy()

    state_data["contact_stall_start_time"] = None
    state_data["stall_start_time"] = None
    state_data["hole_found_start_time"] = None

    if new_state == STATE_LIFT:
        current_grasp_center = d.site("grasp_center").xpos.copy()
        state_data["lift_target"] = current_grasp_center + np.array([0.0, 0.0, 0.08])

    elif new_state == STATE_RELEASE:
        state_data["release_target"] = d.site("grasp_center").xpos.copy()
        state_data["release_open_started"] = False

    elif new_state == STATE_RETREAT:
        current_grasp_center = d.site("grasp_center").xpos.copy()
        state_data["retreat_target"] = current_grasp_center + np.array([0.0, 0.0, 0.05])


def clamp_down_target(start_pos, target_xy, min_z, down_speed, t):
    p = start_pos.copy()
    p[0] = target_xy[0]
    p[1] = target_xy[1]
    p[2] = max(min_z, start_pos[2] - down_speed * t)
    return p


def compute_tcp_velocity(curr_pos, prev_pos, dt):
    if dt <= 1e-6:
        return np.zeros(3)
    return (curr_pos - prev_pos) / dt


def torque_effort_norm(state_data):
    return np.linalg.norm(state_data["last_tau_cmd"])


def maybe_lock_peg(m, d, state_data):
    if state_data["grasped"]:
        return

    grasp_center = d.site("grasp_center").xpos.copy()

    dx = grasp_center[0] - state_data["peg_grasp_target"][0]
    dy = grasp_center[1] - state_data["peg_grasp_target"][1]

    left_count, left_normal, left_total = get_contact_pair_force(
        m, d,
        state_data["left_finger_geom_id"],
        state_data["peg_geom_id"]
    )

    right_count, right_normal, right_total = get_contact_pair_force(
        m, d,
        state_data["right_finger_geom_id"],
        state_data["peg_geom_id"]
    )

    state_data["left_contact_count"] = left_count
    state_data["right_contact_count"] = right_count
    state_data["left_contact_normal"] = left_normal
    state_data["right_contact_normal"] = right_normal
    state_data["left_contact_total"] = left_total
    state_data["right_contact_total"] = right_total

    both_sides_touching = (left_count > 0) and (right_count > 0)
    enough_force = (
        left_normal >= state_data["grasp_normal_force_threshold"] and
        right_normal >= state_data["grasp_normal_force_threshold"]
    )
    xy_aligned = (
        abs(dx) < state_data["grasp_xy_tol"] and
        abs(dy) < state_data["grasp_xy_tol"]
    )

    success_candidate = both_sides_touching and enough_force and xy_aligned

    if success_candidate:
        if state_data["contact_hold_start"] is None:
            state_data["contact_hold_start"] = state_data["time_now"]

        hold_time = state_data["time_now"] - state_data["contact_hold_start"]

        if hold_time >= state_data["grasp_contact_hold_time"]:
            state_data["grasped"] = True
            state_data["grasp_success_time"] = state_data["time_now"]
            state_data["grasp_hold_target"] = grasp_center.copy()

            print(
                f"[GRASP] peg locked | "
                f"Lcnt={left_count}, Rcnt={right_count}, "
                f"Lfn={left_normal:.3f}, Rfn={right_normal:.3f}, "
                f"dx={dx:.6f}, dy={dy:.6f}"
            )
    else:
        state_data["contact_hold_start"] = None


def hole_search_wiggle_offsets(t, roll_deg=2.0, pitch_deg=2.0, freq=1.5, xy_amp=0.0003):
    w = 2.0 * np.pi * freq

    droll = roll_deg * np.sin(w * t)
    dpitch = pitch_deg * np.sin(w * t + np.pi / 2.0)

    dx = xy_amp * np.sin(w * t)
    dy = xy_amp * np.sin(w * t + np.pi / 2.0)

    return dx, dy, droll, dpitch


def get_target_by_state(state_data):
    base_rpy = np.array([90.0, 0.0, 0.0])

    peg_above = state_data["peg_above_target"]
    peg_pregrasp = state_data["peg_pregrasp_target"]
    peg_grasp = state_data["peg_grasp_target"]

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    offset = state_data["grasp_to_peg_tip_offset_nominal_world"]

    state = state_data["state"]
    tau = state_data["time_now"] - state_data["state_enter_time"]

    entry_target = hole_entry + offset
    bottom_target = hole_bottom + offset
    preinsert_target = hole_preinsert + offset + np.array([0.0, 0.0, state_data["preinsert_extra_z"]])

    desired_rpy = base_rpy.copy()

    if state == STATE_APPROACH_ABOVE_PEG:
        desired_xpos_site = peg_above
        control_site_name = "grasp_center"
        gripper_cmd = 0.012

    elif state == STATE_APPROACH_DOWN_TO_PEG:
        desired_xpos_site = peg_pregrasp
        control_site_name = "grasp_center"
        gripper_cmd = 0.012

    elif state == STATE_GRASP:
        if state_data["grasped"]:
            desired_xpos_site = state_data["grasp_hold_target"].copy()
        else:
            desired_xpos_site = peg_grasp

        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_LIFT:
        desired_xpos_site = state_data["lift_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_MOVE_PREINSERT:
        desired_xpos_site = preinsert_target
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_CONTACT_APPROACH:
        desired_xpos_site = clamp_down_target(
            state_data["phase_start_pos"],
            entry_target[:2],
            entry_target[2],
            state_data["touch_down_speed"],
            tau
        )
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_HOLE_SEARCH:
        dx, dy, droll, dpitch = hole_search_wiggle_offsets(
            tau,
            roll_deg=state_data["hole_search_wiggle_roll_deg"],
            pitch_deg=state_data["hole_search_wiggle_pitch_deg"],
            freq=state_data["hole_search_wiggle_freq"],
            xy_amp=state_data["hole_search_xy_amp"]
        )

        desired_xpos_site = entry_target.copy()
        desired_xpos_site[0] += dx
        desired_xpos_site[1] += dy
        desired_xpos_site[2] = max(
            bottom_target[2],
            state_data["phase_start_pos"][2] - state_data["hole_search_down_speed"] * tau
        )

        desired_rpy = base_rpy + np.array([droll, dpitch, 0.0])
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_WIGGLE:
        desired_xpos_site = entry_target.copy()
        desired_xpos_site[2] = max(
            bottom_target[2],
            state_data["phase_start_pos"][2] - state_data["hole_search_down_speed"] * tau
        )
        desired_rpy = base_rpy + np.array([
            state_data["wiggle_amp_deg"] * np.sin(2.0 * np.pi * state_data["wiggle_freq"] * tau),
            state_data["wiggle_amp_deg"] * np.cos(2.0 * np.pi * state_data["wiggle_freq"] * tau),
            0.0
        ])
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_SCREW:
        desired_xpos_site = entry_target.copy()
        desired_xpos_site[2] = max(
            bottom_target[2],
            state_data["phase_start_pos"][2] - state_data["hole_search_down_speed"] * tau
        )
        desired_rpy = base_rpy + np.array([
            0.0,
            0.0,
            state_data["screw_amp_deg"] * np.sin(2.0 * np.pi * state_data["screw_freq"] * tau)
        ])
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_INSERT_FINAL:
        desired_xpos_site = bottom_target.copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.000

    elif state == STATE_RELEASE:
        desired_xpos_site = state_data["release_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.012

    elif state == STATE_RETREAT:
        desired_xpos_site = state_data["retreat_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.012

    else:
        desired_xpos_site = state_data["retreat_target"].copy()
        control_site_name = "grasp_center"
        gripper_cmd = 0.012

    return desired_xpos_site, desired_rpy, gripper_cmd, control_site_name


def update_state(m, d, state_data):
    state = state_data["state"]
    now = state_data["time_now"]
    tau = now - state_data["state_enter_time"]

    grasp_center = d.site("grasp_center").xpos.copy()
    prev_grasp_center = state_data["prev_grasp_center"].copy()

    peg_above = state_data["peg_above_target"]
    peg_pregrasp = state_data["peg_pregrasp_target"]

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    offset = state_data["grasp_to_peg_tip_offset_nominal_world"]

    entry_target = hole_entry + offset
    bottom_target = hole_bottom + offset
    preinsert_target = hole_preinsert + offset + np.array([0.0, 0.0, state_data["preinsert_extra_z"]])

    dt = max(1e-6, now - state_data["prev_time"])
    tcp_vel = compute_tcp_velocity(grasp_center, prev_grasp_center, dt)
    vz_down = -tcp_vel[2]   # 아래로 갈수록 +
    vxy = np.linalg.norm(tcp_vel[:2])
    effort = torque_effort_norm(state_data)

    if state == STATE_APPROACH_ABOVE_PEG:
        err = np.linalg.norm(grasp_center - peg_above)
        if err < 0.004:
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
            wait_time = now - state_data["grasp_success_time"]
            if wait_time > state_data["grasp_wait_time"]:
                print("[STATE] GRASP -> LIFT")
                set_state(STATE_LIFT, state_data, d, now)

    elif state == STATE_LIFT:
        err_vec = grasp_center - state_data["lift_target"]
        err_xy = np.linalg.norm(err_vec[:2])
        err_z = abs(err_vec[2])

        if err_z < 0.005 and err_xy < 0.010:
            print("[STATE] LIFT -> MOVE_PREINSERT")
            set_state(STATE_MOVE_PREINSERT, state_data, d, now)

    elif state == STATE_MOVE_PREINSERT:
        err = np.linalg.norm(grasp_center - preinsert_target)
        if err < 0.006:
            print("[STATE] MOVE_PREINSERT -> CONTACT_APPROACH")
            set_state(STATE_CONTACT_APPROACH, state_data, d, now)

    elif state == STATE_CONTACT_APPROACH:
        contact_stuck = (
            tau > state_data["contact_min_time"] and
            vz_down < state_data["contact_vz_thresh"] and
            effort > state_data["contact_effort_thresh"]
        )

        if contact_stuck:
            if state_data["contact_stall_start_time"] is None:
                state_data["contact_stall_start_time"] = now
        else:
            state_data["contact_stall_start_time"] = None

        if (
            state_data["contact_stall_start_time"] is not None and
            now - state_data["contact_stall_start_time"] > state_data["contact_hold_time"]
        ):
            print("[STATE] CONTACT_APPROACH -> HOLE_SEARCH")
            set_state(STATE_HOLE_SEARCH, state_data, d, now)

    elif state == STATE_HOLE_SEARCH:
        hole_found = (vz_down > state_data["hole_found_vz_thresh"])

        if hole_found:
            if state_data["hole_found_start_time"] is None:
                state_data["hole_found_start_time"] = now
        else:
            state_data["hole_found_start_time"] = None

        if (
            state_data["hole_found_start_time"] is not None and
            now - state_data["hole_found_start_time"] > state_data["hole_found_hold_time"]
        ):
            print("[STATE] HOLE_SEARCH -> INSERT_FINAL")
            set_state(STATE_INSERT_FINAL, state_data, d, now)

        elif tau > state_data["hole_search_timeout"]:
            print("[STATE] HOLE_SEARCH -> RETREAT (timeout)")
            set_state(STATE_RETREAT, state_data, d, now)

    elif state == STATE_WIGGLE:
        if tau > state_data["wiggle_time"]:
            print("[STATE] WIGGLE -> SCREW")
            set_state(STATE_SCREW, state_data, d, now)

    elif state == STATE_SCREW:
        if tau > state_data["screw_time"]:
            print("[STATE] SCREW -> INSERT_FINAL")
            set_state(STATE_INSERT_FINAL, state_data, d, now)

    elif state == STATE_INSERT_FINAL:
        err = np.linalg.norm(grasp_center - bottom_target)
        if err < 0.004:
            print("[STATE] INSERT_FINAL -> RELEASE")
            set_state(STATE_RELEASE, state_data, d, now)

    elif state == STATE_RELEASE:
        time_in_state = now - state_data["state_enter_time"]

        if (not state_data["release_open_started"]) and time_in_state > state_data["release_ungrasp_time"]:
            state_data["grasped"] = False
            state_data["release_open_started"] = True
            print("[STATE] RELEASE | peg detached")

        if time_in_state > state_data["release_total_time"]:
            print("[STATE] RELEASE -> RETREAT")
            set_state(STATE_RETREAT, state_data, d, now)

    elif state == STATE_RETREAT:
        err = np.linalg.norm(grasp_center - state_data["retreat_target"])
        if err < 0.005:
            print("[STATE] RETREAT -> DONE")
            set_state(STATE_DONE, state_data, d, now)

    state_data["prev_grasp_center"] = grasp_center.copy()


def print_debug(d, state_data, desired_xpos_site, control_site_name, print_every=0.1):
    now = state_data["time_now"]
    if now - state_data["last_print_time"] < print_every:
        return
    state_data["last_print_time"] = now

    current_pos = d.site(control_site_name).xpos.copy()
    pos_err = current_pos - desired_xpos_site
    tau_norm = np.linalg.norm(state_data["last_tau_cmd"])

    print(
        f"[{get_state_name(state_data['state'])} | site={control_site_name}] "
        f"POS=({current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}) | "
        f"ERR=({pos_err[0]:.4f}, {pos_err[1]:.4f}, {pos_err[2]:.4f}) | "
        f"grasped={state_data['grasped']} | "
        f"gcmd={state_data['gripper_cmd_current']:.4f} | "
        f"tau={tau_norm:.3f} | "
        f"Lcnt={state_data['left_contact_count']} "
        f"Rcnt={state_data['right_contact_count']} "
        f"Lfn={state_data['left_contact_normal']:.3f} "
        f"Rfn={state_data['right_contact_normal']:.3f}"
    )


def resolve_model_path():
    local_path = Path(__file__).resolve().parent / "scene_rb5_gripper.xml"
    if local_path.exists():
        return str(local_path)

    return "/home/chu/manipulator_control/rb5/pick_and_place/scene_rb5_gripper.xml"


def main():
    model_path = resolve_model_path()
    m, d = create_model_and_data(model_path)

    validate_required_sites(m)
    validate_required_geoms(m)
    initialize_robot_state(m, d)

    M, G, jacp, jacr, C0 = create_work_buffers(m)
    peg_qpos_adr, peg_qvel_adr = get_peg_freejoint_addrs(m)

    # 게인
    K_a = np.array([60.0, 60.0, 35.0])
    zeta_a = np.array([8.0, 8.0, 4.0])

    K_o = np.array([3.0, 3.0, 1.5])
    zeta_o = np.array([0.3, 0.3, 0.2])

    mujoco.mj_forward(m, d)

    # -----------------------------
    # 초기 인식값(한 번만 저장)
    # -----------------------------
    peg_init_body_world = d.body("peg").xpos.copy()
    peg_init_center_world = d.site("peg_center").xpos.copy()
    peg_init_tip_world = d.site("peg_tip").xpos.copy()

    hole_preinsert_world = d.site("hole_preinsert").xpos.copy()
    hole_entry_world = d.site("hole_entry").xpos.copy()
    hole_bottom_world = d.site("hole_bottom").xpos.copy()

    peg_init_quat_world = d.qpos[peg_qpos_adr+3:peg_qpos_adr+7].copy()

    peg_above_target = peg_init_center_world + np.array([0.0, 0.0, 0.05])
    peg_pregrasp_target = peg_init_center_world + np.array([0.0, 0.0, 0.013])
    peg_grasp_target = peg_init_center_world + np.array([0.0, 0.0, 0.013])

    grasp_to_peg_tip_offset_nominal_world = peg_grasp_target - peg_init_tip_world
    peg_body_offset_from_grasp_nominal_world = peg_init_body_world - peg_grasp_target

    initial_grasp_center = d.site("grasp_center").xpos.copy()

    state_data = {
        "state": STATE_APPROACH_ABOVE_PEG,
        "state_enter_time": 0.0,
        "time_now": 0.0,
        "last_print_time": -1.0,
        "prev_time": 0.0,

        "grasped": False,
        "grasp_success_time": None,

        "lift_target": np.zeros(3),
        "release_target": np.zeros(3),
        "retreat_target": np.zeros(3),
        "grasp_hold_target": np.zeros(3),

        "peg_qpos_adr": peg_qpos_adr,
        "peg_qvel_adr": peg_qvel_adr,

        "peg_init_quat_world": peg_init_quat_world,
        "peg_body_offset_from_grasp_nominal_world": peg_body_offset_from_grasp_nominal_world,
        "grasp_to_peg_tip_offset_nominal_world": grasp_to_peg_tip_offset_nominal_world,

        "peg_above_target": peg_above_target,
        "peg_pregrasp_target": peg_pregrasp_target,
        "peg_grasp_target": peg_grasp_target,

        "hole_preinsert_world": hole_preinsert_world,
        "hole_entry_world": hole_entry_world,
        "hole_bottom_world": hole_bottom_world,

        "left_finger_geom_id": m.geom("left_finger_geom").id,
        "right_finger_geom_id": m.geom("right_finger_geom").id,
        "peg_geom_id": m.geom("peg_geom").id,

        "contact_hold_start": None,

        "grasp_normal_force_threshold": 0.5,
        "grasp_contact_hold_time": 0.05,
        "grasp_xy_tol": 0.0035,
        "grasp_wait_time": 2.0,

        "left_contact_count": 0,
        "right_contact_count": 0,
        "left_contact_normal": 0.0,
        "right_contact_normal": 0.0,
        "left_contact_total": 0.0,
        "right_contact_total": 0.0,

        "gripper_act_id": m.actuator("gripper_act").id,
        "gripper_cmd_current": 0.010,
        "gripper_close_speed": 0.015,
        "gripper_open_speed": 0.030,

        "release_open_started": False,
        "release_ungrasp_time": 0.20,
        "release_total_time": 0.50,

        "preinsert_extra_z": 0.03,

        "touch_down_speed": 0.010,

        # contact 후 hole-search(wiggle search)
        "hole_search_down_speed": 0.0010,
        "hole_search_wiggle_roll_deg": 2.0,
        "hole_search_wiggle_pitch_deg": 2.0,
        "hole_search_wiggle_freq": 1.5,
        "hole_search_xy_amp": 0.0003,
        "hole_search_timeout": 1.5,

        # 아래 두 상태는 현재 FSM에서 직접 사용하지 않지만 남겨둠
        "wiggle_amp_deg": 3.0,
        "wiggle_freq": 2.0,
        "wiggle_time": 1.0,

        "screw_amp_deg": 6.0,
        "screw_freq": 1.5,
        "screw_time": 1.0,

        # CONTACT_APPROACH 판단
        "contact_stall_start_time": None,
        "contact_min_time": 0.20,
        "contact_vz_thresh": 0.0010,
        "contact_effort_thresh": 12.0,
        "contact_hold_time": 0.12,

        # HOLE_SEARCH 판단
        "stall_start_time": None,
        "hole_found_start_time": None,

        "hole_found_vz_thresh": 0.0020,
        "hole_found_hold_time": 0.08,

        "phase_start_pos": initial_grasp_center.copy(),
        "prev_grasp_center": initial_grasp_center.copy(),

        "last_tau_cmd": np.zeros(6),
    }

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time()

        while viewer.is_running():
            now = time() - t0
            state_data["time_now"] = now

            desired_xpos_site, desired_rpy, gripper_cmd, control_site_name = get_target_by_state(
                state_data
            )

            compute_mass_and_gravity(m, d, M, G)

            torque0 = compute_task_torque(
                m, d,
                M, G, jacp, jacr, C0,
                desired_xpos_site, desired_rpy,
                K_a, zeta_a, K_o, zeta_o,
                site_name=control_site_name
            )

            apply_control(
                m, d, torque0, state_data,
                max_torque=80,
                gripper_cmd_target=gripper_cmd
            )

            state_data["last_tau_cmd"] = d.ctrl[0:6].copy()

            # sim 편의용 peg attach
            if state_data["grasped"]:
                attach_peg_to_gripper(m, d, state_data)
                mujoco.mj_forward(m, d)

            mujoco.mj_step(m, d)

            print_debug(d, state_data, desired_xpos_site, control_site_name, print_every=0.1)
            update_state(m, d, state_data)

            viewer.sync()
            state_data["prev_time"] = now


if __name__ == "__main__":
    main()