from time import time
from copy import deepcopy
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# =========================
# User-tunable geometry
# =========================
PEG_LENGTH_M = 0.050          # 50 mm
HOLE_TOP_HEIGHT_M = 0.040     # 40 mm  (hole bottom 기준 top까지 높이)


CONTACT_TO_SEARCH_Z_MARGIN_M = 0.000  # 필요시 0.001 정도로 조정
CONTACT_STALL_TOP_BAND_M = 0.005      # hole top ±5 mm
CONTACT_STALL_V_THRESH_MPS = 0.0005   # 거의 못 내려감 판단 기준
CONTACT_BELOW_TOP_CHECK_M = 0.010     # hole top보다 10 mm 아래
CONTACT_RETREAT_XY_THRESH_M = 0.020   # 20 mm

CONTACT_PUSH_BELLOW_TOP_M = 0.020     # CONTACT_APPROACH에서 top보다 20 mm 아래까지 계속 밀 목표
RETREAT_LIFT_M = 0.050                # RETREAT 시 위로 50 mm


STATE_MOVE_PREINSERT = 0
STATE_CONTACT_APPROACH = 1
STATE_HOLE_SEARCH = 2
STATE_INSERT_FINAL = 3
STATE_RETREAT = 4
STATE_DONE = 5



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
        "peg_base",
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
            f"rb5_peg_in_hole.xml / scene_rb5_peg_in_hole.xml의 site 이름을 확인하세요."
        )


def initialize_robot_state(m, d):
    # arm home
    d.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
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
    site_name="peg_tip",
    pos_task_axis_mask=None
):
    if pos_task_axis_mask is None:
        pos_task_axis_mask = np.ones(3, dtype=np.float64)
    else:
        pos_task_axis_mask = np.asarray(pos_task_axis_mask, dtype=np.float64)

    np.fill_diagonal(C0, np.sum(np.abs(M[0:6, 0:6]), axis=1))

    site_id = m.site(site_name).id
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)

    jacp0 = deepcopy(jacp[:, 0:6])
    jacr0 = deepcopy(jacr[:, 0:6])

    xpos_err0 = d.site(site_name).xpos - desired_xpos_site
    xpos_err0 = xpos_err0 * pos_task_axis_mask

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
    xpos_dot0 = xpos_dot0 * pos_task_axis_mask

    force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

    M0 = M[0:6, 0:6]
    M0_inv = np.linalg.pinv(M0)

    eps = 1e-6
    Lambda_inv = jacp0 @ M0_inv @ jacp0.T
    Lambda_pos = np.linalg.pinv(Lambda_inv + eps * np.eye(3))

    force0_task = Lambda_pos @ force0
    force0_task = force0_task * pos_task_axis_mask

    torque0 = (
        -0 * C0 @ d.qvel[0:6]
        -1 * jacp0.T @ force0_task
        +1 * G[0:6]
        -1 * jacr0.T @ F_ori_0
    )

    return torque0


def apply_control(d, torque0, max_torque=80):
    d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)


def get_state_name(state):
    if state == STATE_MOVE_PREINSERT:
        return "MOVE_PREINSERT"
    elif state == STATE_CONTACT_APPROACH:
        return "CONTACT_APPROACH"
    elif state == STATE_HOLE_SEARCH:
        return "HOLE_SEARCH"
    elif state == STATE_INSERT_FINAL:
        return "INSERT_FINAL"
    elif state == STATE_RETREAT:
        return "RETREAT"
    else:
        return "DONE"


def clamp_down_target(start_pos, target_xy, min_z, down_speed, t):
    p = start_pos.copy()
    p[0] = target_xy[0]
    p[1] = target_xy[1]
    p[2] = max(min_z, start_pos[2] - down_speed * t)
    return p


def compute_velocity(curr_pos, prev_pos, dt):
    if dt <= 1e-6:
        return np.zeros(3)
    return (curr_pos - prev_pos) / dt


def torque_effort_norm(state_data):
    return np.linalg.norm(state_data["last_tau_cmd"])


def hole_search_wiggle_offsets(t, roll_deg=1.0, pitch_deg=1.0, freq=1.0, xy_amp=0.0):
    w = 2.0 * np.pi * freq

    droll = roll_deg * np.sin(w * t)
    dpitch = pitch_deg * np.sin(w * t + np.pi / 2.0)

    dx = xy_amp * np.sin(w * t)
    dy = xy_amp * np.sin(w * t + np.pi / 2.0)

    return dx, dy, droll, dpitch


def set_state(new_state, state_data, d, now):
    state_data["state"] = new_state
    state_data["state_enter_time"] = now
    state_data["phase_start_pos"] = d.site("peg_tip").xpos.copy()
    state_data["hole_found_start_time"] = None
    state_data["insert_final_stall_start_time"] = None

    if new_state == STATE_RETREAT:
        current_tip = d.site("peg_tip").xpos.copy()
        state_data["retreat_target"] = current_tip + np.array([0.0, 0.0, state_data["retreat_lift_m"]])

    elif new_state == STATE_DONE:
        state_data["done_hold_target"] = d.site("peg_tip").xpos.copy()


def get_target_by_state(state_data):
    base_rpy = np.array([90.0, 0.0, 0.0])

    state = state_data["state"]
    tau = state_data["time_now"] - state_data["state_enter_time"]

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    desired_rpy = base_rpy.copy()
    control_site_name = "peg_tip"

    hole_top_z = hole_bottom[2] + state_data["hole_top_height_m"]

    if state == STATE_MOVE_PREINSERT:
        desired_xpos_site = hole_preinsert.copy()

    elif state == STATE_CONTACT_APPROACH:
        # x,y는 hole 중심에 유지하고, z는 계속 -z 방향으로 밀어 넣음
        contact_min_z = hole_top_z - state_data["contact_push_below_top_m"]

        desired_xpos_site = clamp_down_target(
            state_data["phase_start_pos"],
            hole_entry[:2],
            contact_min_z,
            state_data["touch_down_speed"],
            tau
        )

    elif state == STATE_HOLE_SEARCH:
        dx, dy, droll, dpitch = hole_search_wiggle_offsets(
            tau,
            roll_deg=state_data["hole_search_wiggle_roll_deg"],
            pitch_deg=state_data["hole_search_wiggle_pitch_deg"],
            freq=state_data["hole_search_wiggle_freq"],
            xy_amp=state_data["hole_search_xy_amp"]
        )

        desired_xpos_site = hole_entry.copy()
        desired_xpos_site[0] += dx
        desired_xpos_site[1] += dy
        desired_xpos_site[2] = max(
            hole_bottom[2],
            state_data["phase_start_pos"][2] - state_data["hole_search_down_speed"] * tau
        )

        desired_rpy = base_rpy + np.array([droll, dpitch, 0.0])

    elif state == STATE_INSERT_FINAL:
        desired_xpos_site = state_data["phase_start_pos"].copy()
        desired_xpos_site[2] = max(
            state_data["insert_final_target_z_m"],
            state_data["phase_start_pos"][2] - state_data["insert_final_down_speed"] * tau
        )

    elif state == STATE_RETREAT:
        desired_xpos_site = state_data["retreat_target"].copy()

    elif state == STATE_DONE:
        desired_xpos_site = state_data["done_hold_target"].copy()

    else:
        desired_xpos_site = state_data["done_hold_target"].copy()

    return desired_xpos_site, desired_rpy, control_site_name


def update_state(m, d, state_data):
    state = state_data["state"]
    now = state_data["time_now"]
    tau = now - state_data["state_enter_time"]

    peg_tip = d.site("peg_tip").xpos.copy()
    prev_peg_tip = state_data["prev_peg_tip"].copy()

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    dt = max(1e-6, now - state_data["prev_time"])
    tip_vel = compute_velocity(peg_tip, prev_peg_tip, dt)

    tip_vz_down = -tip_vel[2]          # 아래 방향이면 +
    tip_vz_abs = abs(tip_vel[2])
    tip_xy_err_entry = np.linalg.norm((peg_tip - hole_entry)[:2])
    tip_xy_err_bottom = np.linalg.norm((peg_tip - hole_bottom)[:2])

    effort = torque_effort_norm(state_data)

    hole_top_z = hole_bottom[2] + state_data["hole_top_height_m"]
    near_hole_top = abs(peg_tip[2] - hole_top_z) <= state_data["contact_stall_top_band_m"]
    below_hole_top_10mm = peg_tip[2] <= (hole_top_z - state_data["contact_below_top_check_m"])

    if state == STATE_MOVE_PREINSERT:
        err = np.linalg.norm(peg_tip - hole_preinsert)
        if err < state_data["preinsert_tol"]:
            print("[STATE] MOVE_PREINSERT -> CONTACT_APPROACH")
            set_state(STATE_CONTACT_APPROACH, state_data, d, now)

    elif state == STATE_CONTACT_APPROACH:
        # 1) hole top 근처(±5 mm)에서 거의 못 내려가면 HOLE_SEARCH
        if near_hole_top and tip_vz_abs < state_data["contact_stall_v_thresh_mps"]:
            print("[STATE] CONTACT_APPROACH -> HOLE_SEARCH")
            set_state(STATE_HOLE_SEARCH, state_data, d, now)

        # 2) hole top보다 10 mm 더 아래까지 내려갔을 때 분기
        elif below_hole_top_10mm:
            if tip_xy_err_entry > state_data["contact_retreat_xy_thresh"]:
                print("[STATE] CONTACT_APPROACH -> RETREAT (xy too far below hole top)")
                set_state(STATE_RETREAT, state_data, d, now)
            else:
                print("[STATE] CONTACT_APPROACH -> INSERT_FINAL")
                set_state(STATE_INSERT_FINAL, state_data, d, now)

    elif state == STATE_HOLE_SEARCH:
        hole_found = (
            tip_vz_down > state_data["hole_found_vz_thresh"] and
            tip_xy_err_entry < state_data["hole_found_xy_thresh"] and
            peg_tip[2] < (hole_entry[2] - state_data["hole_found_z_margin"])
        )

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

    elif state == STATE_INSERT_FINAL:
        # DONE 조건: peg tip z가 지면에서 5 mm 이하 도달
        if peg_tip[2] <= state_data["insert_final_done_z_m"]:
            print("[STATE] INSERT_FINAL -> DONE")
            set_state(STATE_DONE, state_data, d, now)

        else:
            # stall 감시는 z > 8 mm 구간에서만 수행
            monitor_zone = peg_tip[2] > state_data["insert_final_stall_monitor_z_m"]

            # "v = 0"는 실제론 exact 0보다 threshold로 보는 게 안정적
            stalled = tip_vz_abs < state_data["insert_final_stall_v_thresh_mps"]

            if monitor_zone and stalled:
                if state_data["insert_final_stall_start_time"] is None:
                    state_data["insert_final_stall_start_time"] = now
                elif (now - state_data["insert_final_stall_start_time"]) > state_data["insert_final_stall_hold_time"]:
                    print("[STATE] INSERT_FINAL -> RETREAT (timeout)")
                    set_state(STATE_RETREAT, state_data, d, now)
            else:
                state_data["insert_final_stall_start_time"] = None

    elif state == STATE_RETREAT:
        err = np.linalg.norm(peg_tip - state_data["retreat_target"])

        if err < state_data["retreat_tol"]:
            print("[STATE] RETREAT -> MOVE_PREINSERT")
            set_state(STATE_MOVE_PREINSERT, state_data, d, now)

    state_data["prev_peg_tip"] = peg_tip.copy()


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
        f"tau={tau_norm:.3f}"
    )


def resolve_model_path():
    local_path = Path(__file__).resolve().parent / "scene_rb5_peg_in_hole.xml"
    if local_path.exists():
        return str(local_path)

    return "/home/chu/manipulator_control/rb5/peg_in_hole/scene_rb5_peg_in_hole.xml"


def main():
    model_path = resolve_model_path()
    m, d = create_model_and_data(model_path)

    validate_required_sites(m)
    initialize_robot_state(m, d)

    M, G, jacp, jacr, C0 = create_work_buffers(m)

    # 게인
    K_a = np.array([60.0, 60.0, 35.0])
    zeta_a = np.array([8.0, 8.0, 4.0])

    K_o = np.array([3.0, 3.0, 1.5])
    zeta_o = np.array([0.3, 0.3, 0.2])

    mujoco.mj_forward(m, d)

    hole_preinsert_world = d.site("hole_preinsert").xpos.copy()
    hole_entry_world = d.site("hole_entry").xpos.copy()
    hole_bottom_world = d.site("hole_bottom").xpos.copy()

    initial_peg_tip = d.site("peg_tip").xpos.copy()

    state_data = {
        "state": STATE_MOVE_PREINSERT,
        "state_enter_time": 0.0,
        "time_now": 0.0,
        "last_print_time": -1.0,
        "prev_time": 0.0,

        "hole_preinsert_world": hole_preinsert_world,
        "hole_entry_world": hole_entry_world,
        "hole_bottom_world": hole_bottom_world,

        "retreat_target": initial_peg_tip.copy(),
        "done_hold_target": initial_peg_tip.copy(),

        "preinsert_tol": 0.004,

        "touch_down_speed": 0.010,
        "contact_xy_tol": 0.0015,
        "contact_z_tol": 0.0030,
        "contact_approach_timeout": 3.0,

        "hole_search_down_speed": 0.0005,
        "hole_search_wiggle_roll_deg": 1.0,
        "hole_search_wiggle_pitch_deg": 1.0,
        "hole_search_wiggle_freq": 1.0,
        "hole_search_xy_amp": 0.0,
        "hole_search_timeout": 1.5,

        "hole_found_start_time": None,
        "hole_found_vz_thresh": 0.0015,
        "hole_found_xy_thresh": 0.0010,
        "hole_found_z_margin": 0.0005,
        "hole_found_hold_time": 0.08,

        "insert_xy_tol": 0.0010,
        "insert_z_tol": 0.0020,
        "insert_timeout": 3.0,
        "insert_final_down_speed": 0.008,

        "retreat_tol": 0.005,

        "phase_start_pos": initial_peg_tip.copy(),
        "prev_peg_tip": initial_peg_tip.copy(),

        "last_tau_cmd": np.zeros(6),

        "contact_retreat_xy_thresh": 0.010,   # 10 mm
        "contact_to_search_z_margin": 0.0000, # 필요하면 0.001 정도로 줄 수 있음

        "peg_length_m": PEG_LENGTH_M,
        "hole_top_height_m": HOLE_TOP_HEIGHT_M,
        "contact_retreat_xy_thresh": CONTACT_RETREAT_XY_THRESH_M,
        "contact_to_search_z_margin": CONTACT_TO_SEARCH_Z_MARGIN_M,

                "peg_length_m": PEG_LENGTH_M,
        "hole_top_height_m": HOLE_TOP_HEIGHT_M,

        "contact_stall_top_band_m": CONTACT_STALL_TOP_BAND_M,
        "contact_stall_v_thresh_mps": CONTACT_STALL_V_THRESH_MPS,
        "contact_below_top_check_m": CONTACT_BELOW_TOP_CHECK_M,
        "contact_retreat_xy_thresh": CONTACT_RETREAT_XY_THRESH_M,

        "contact_push_below_top_m": CONTACT_PUSH_BELLOW_TOP_M,
        "retreat_lift_m": RETREAT_LIFT_M,

        "insert_final_down_speed": 0.010,
        "insert_final_target_z_m": 0.005,
        "insert_final_stall_monitor_z_m": 0.008,
        "insert_final_stall_v_thresh_mps": 0.0005,
        "insert_final_stall_hold_time": 2.0,
        "insert_final_stall_start_time": None,
        "insert_final_done_z_m": 0.006,
    }

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time()

        while viewer.is_running():
            now = time() - t0
            state_data["time_now"] = now

            desired_xpos_site, desired_rpy, control_site_name = get_target_by_state(state_data)

            compute_mass_and_gravity(m, d, M, G)

            if state_data["state"] == STATE_INSERT_FINAL:
                torque0 = compute_task_torque(
                    m, d,
                    M, G, jacp, jacr, C0,
                    desired_xpos_site, desired_rpy,
                    K_a, zeta_a, K_o, zeta_o,
                    site_name=control_site_name,
                    pos_task_axis_mask=np.array([0.0, 0.0, 1.0])
                )
            else:
                torque0 = compute_task_torque(
                    m, d,
                    M, G, jacp, jacr, C0,
                    desired_xpos_site, desired_rpy,
                    K_a, zeta_a, K_o, zeta_o,
                    site_name=control_site_name
                )

            apply_control(d, torque0, max_torque=80)
            state_data["last_tau_cmd"] = d.ctrl[0:6].copy()

            mujoco.mj_step(m, d)

            print_debug(d, state_data, desired_xpos_site, control_site_name, print_every=0.1)
            update_state(m, d, state_data)

            viewer.sync()
            state_data["prev_time"] = now


if __name__ == "__main__":
    main()