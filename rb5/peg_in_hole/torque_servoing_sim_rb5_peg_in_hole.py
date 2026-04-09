from time import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# =========================
# User-tunable geometry / thresholds
# =========================
TOTAL_TRIALS = 100

HOLE_TOP_HEIGHT_M = 0.040       # hole bottom 기준 top까지 높이 (40 mm)

CONTACT_STALL_TOP_BAND_M = 0.005      # hole top ±5 mm
CONTACT_STALL_V_THRESH_MPS = 0.0005   # 거의 못 내려감 판단 기준
CONTACT_BELOW_TOP_CHECK_M = 0.010     # hole top보다 10 mm 아래
CONTACT_RETREAT_XY_THRESH_M = 0.020   # 20 mm
CONTACT_PUSH_BELOW_TOP_M = 0.020      # CONTACT_APPROACH에서 top보다 20 mm 아래까지 계속 밀 목표

RETREAT_LIFT_M = 0.050                # RETREAT 시 위로 50 mm


STATE_MOVE_PREINSERT = 0
STATE_CONTACT_APPROACH = 1
STATE_HOLE_SEARCH = 2
STATE_INSERT_FINAL = 3
STATE_RETREAT = 4
STATE_DONE = 5


# =========================
# Target randomization
# =========================
RANDOMIZE_TARGET_XY = True
TARGET_RANDOM_RADIUS_MM = 10.0
RANDOM_SEED = None


# =========================
# Viewer
# =========================
VIEW_HOLE = True





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

    rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])
    ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])
    rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    return rz @ ry @ rx


def create_model_and_data(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def validate_required_sites(model):
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
            _ = model.site(name).id
        except KeyError:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"필수 site가 없습니다: {missing}\n"
            f"rb5_peg_in_hole.xml / scene_rb5_peg_in_hole.xml의 site 이름을 확인하세요."
        )


def initialize_robot_state(model, data):
    _ = model
    data.qpos[:6] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0


def create_work_buffers(model):
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    gravity = np.zeros(model.nv, dtype=np.float64)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    c_joint = np.zeros((6, 6), dtype=np.float64)
    return mass, gravity, jacp, jacr, c_joint


def compute_mass_and_gravity(model, data, mass, gravity):
    mujoco.mj_fullM(model, mass, data.qM)

    qvel_backup = data.qvel.copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_rne(model, data, 0, gravity)
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)


def compute_task_torque(
    model,
    data,
    mass,
    gravity,
    jacp,
    jacr,
    c_joint,
    desired_xpos_site,
    desired_rpy,
    k_pos,
    zeta_pos,
    k_ori,
    zeta_ori,
    site_name="peg_tip",
    pos_task_axis_mask=None,
):
    if pos_task_axis_mask is None:
        pos_task_axis_mask = np.ones(3, dtype=np.float64)
    else:
        pos_task_axis_mask = np.asarray(pos_task_axis_mask, dtype=np.float64)

    np.fill_diagonal(c_joint, np.sum(np.abs(mass[0:6, 0:6]), axis=1))

    site_id = model.site(site_name).id
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    jacp0 = jacp[:, 0:6].copy()
    jacr0 = jacr[:, 0:6].copy()

    xpos_err = data.site(site_name).xpos - desired_xpos_site
    xpos_err = xpos_err * pos_task_axis_mask

    r_current = data.site(site_id).xmat.reshape(3, 3)
    r_desired = rpy_to_rotmat(*desired_rpy)

    ori_err = (
        np.cross(r_desired[:, 0], r_current[:, 0]) +
        np.cross(r_desired[:, 1], r_current[:, 1]) +
        np.cross(r_desired[:, 2], r_current[:, 2])
    )

    w = jacr0 @ data.qvel[0:6]
    f_ori = (k_ori * ori_err) + (zeta_ori * np.sqrt(k_ori) * w)

    xpos_dot = jacp0 @ data.qvel[0:6]
    xpos_dot = xpos_dot * pos_task_axis_mask

    f_pos = (k_pos * xpos_err) + (zeta_pos * np.sqrt(k_pos) * xpos_dot)

    mass0 = mass[0:6, 0:6]
    mass0_inv = np.linalg.pinv(mass0)

    eps = 1e-6
    lambda_inv = jacp0 @ mass0_inv @ jacp0.T
    lambda_pos = np.linalg.pinv(lambda_inv + eps * np.eye(3))

    f_pos_task = lambda_pos @ f_pos
    f_pos_task = f_pos_task * pos_task_axis_mask

    torque = (
        - jacp0.T @ f_pos_task
        + gravity[0:6]
        - jacr0.T @ f_ori
    )
    return torque


def apply_control(data, torque, max_torque=80):
    data.ctrl[0:6] = np.clip(torque, -max_torque, max_torque)


def get_state_name(state):
    names = {
        STATE_MOVE_PREINSERT: "MOVE_PREINSERT",
        STATE_CONTACT_APPROACH: "CONTACT_APPROACH",
        STATE_HOLE_SEARCH: "HOLE_SEARCH",
        STATE_INSERT_FINAL: "INSERT_FINAL",
        STATE_RETREAT: "RETREAT",
        STATE_DONE: "DONE",
    }
    return names.get(state, "UNKNOWN")


def clamp_down_target(start_pos, target_xy, min_z, down_speed, elapsed_time):
    target = start_pos.copy()
    target[0] = target_xy[0]
    target[1] = target_xy[1]
    target[2] = max(min_z, start_pos[2] - down_speed * elapsed_time)
    return target


def compute_velocity(curr_pos, prev_pos, dt):
    if dt <= 1e-6:
        return np.zeros(3, dtype=np.float64)
    return (curr_pos - prev_pos) / dt


def hole_search_wiggle_offsets(
    elapsed_time,
    roll_deg=1.0,
    pitch_deg=1.0,
    freq=1.0,
    xy_amp=0.0,
    decay_time=1.5,
):
    if freq <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    omega = 2.0 * np.pi * freq
    one_cycle_time = 1.0 / freq

    # 한 바퀴 돈 뒤부터 decay 시작
    if elapsed_time <= one_cycle_time:
        amp_scale = 1.0
    else:
        decay_elapsed = elapsed_time - one_cycle_time
        amp_scale = max(0.0, 1.0 - decay_elapsed / decay_time)

    droll = amp_scale * roll_deg * np.sin(omega * elapsed_time)
    dpitch = amp_scale * pitch_deg * np.sin(omega * elapsed_time + np.pi / 2.0)

    dx = amp_scale * xy_amp * np.sin(omega * elapsed_time)
    dy = amp_scale * xy_amp * np.sin(omega * elapsed_time + np.pi / 2.0)

    return dx, dy, droll, dpitch


def sample_random_xy_bias(radius_mm, rng):
    if radius_mm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    radius_m = radius_mm * 1e-3

    # 원 내부 균일 샘플링
    r = radius_m * np.sqrt(rng.uniform(0.0, 1.0))
    theta = rng.uniform(0.0, 2.0 * np.pi)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    return np.array([dx, dy, 0.0], dtype=np.float64)


def build_biased_hole_targets(data, rng):
    hole_preinsert_world = data.site("hole_preinsert").xpos.copy()
    hole_entry_world = data.site("hole_entry").xpos.copy()
    hole_bottom_world = data.site("hole_bottom").xpos.copy()

    target_xy_bias = np.zeros(3, dtype=np.float64)
    if RANDOMIZE_TARGET_XY:
        target_xy_bias = sample_random_xy_bias(TARGET_RANDOM_RADIUS_MM, rng)

    hole_preinsert_world = hole_preinsert_world + target_xy_bias
    hole_entry_world = hole_entry_world + target_xy_bias
    hole_bottom_world = hole_bottom_world + target_xy_bias

    return hole_preinsert_world, hole_entry_world, hole_bottom_world, target_xy_bias


def set_state(new_state, state_data, data, now):
    state_data["state"] = new_state
    state_data["state_enter_time"] = now
    state_data["phase_start_pos"] = data.site("peg_tip").xpos.copy()
    state_data["hole_found_start_time"] = None
    state_data["insert_final_stall_start_time"] = None

    if new_state == STATE_MOVE_PREINSERT:
        state_data["contact_stall_z"] = None

    if new_state == STATE_RETREAT:
        current_tip = data.site("peg_tip").xpos.copy()
        state_data["retreat_target"] = current_tip + np.array([0.0, 0.0, state_data["retreat_lift_m"]])

    elif new_state == STATE_DONE:
        state_data["done_hold_target"] = data.site("peg_tip").xpos.copy()


def get_target_by_state(state_data):
    base_rpy = np.array([90.0, 0.0, 0.0])

    state = state_data["state"]
    elapsed_time = state_data["time_now"] - state_data["state_enter_time"]

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    desired_rpy = base_rpy.copy()
    control_site_name = "peg_tip"

    hole_top_z = hole_bottom[2] + state_data["hole_top_height_m"]

    if state == STATE_MOVE_PREINSERT:
        desired_xpos_site = hole_preinsert.copy()

    elif state == STATE_CONTACT_APPROACH:
        contact_min_z = hole_top_z - state_data["contact_push_below_top_m"]
        desired_xpos_site = clamp_down_target(
            state_data["phase_start_pos"],
            hole_entry[:2],
            contact_min_z,
            state_data["touch_down_speed"],
            elapsed_time,
        )

    elif state == STATE_HOLE_SEARCH:
        dx, dy, droll, dpitch = hole_search_wiggle_offsets(
            elapsed_time,
            roll_deg=state_data["hole_search_wiggle_roll_deg"],
            pitch_deg=state_data["hole_search_wiggle_pitch_deg"],
            freq=state_data["hole_search_wiggle_freq"],
            xy_amp=state_data["hole_search_xy_amp"],
            decay_time=state_data["hole_search_timeout"],
        )

        desired_xpos_site = hole_entry.copy()
        desired_xpos_site[0] += dx
        desired_xpos_site[1] += dy
        desired_xpos_site[2] = max(
            hole_bottom[2],
            state_data["phase_start_pos"][2] - state_data["hole_search_down_speed"] * elapsed_time,
        )

        desired_rpy = base_rpy + np.array([droll, dpitch, 0.0])

    elif state == STATE_INSERT_FINAL:
        desired_xpos_site = state_data["phase_start_pos"].copy()
        desired_xpos_site[2] = max(
            state_data["insert_final_target_z_m"],
            state_data["phase_start_pos"][2] - state_data["insert_final_down_speed"] * elapsed_time,
        )

    elif state == STATE_RETREAT:
        desired_xpos_site = state_data["retreat_target"].copy()

    else:
        desired_xpos_site = state_data["done_hold_target"].copy()

    return desired_xpos_site, desired_rpy, control_site_name


def update_state(data, state_data):
    state = state_data["state"]
    now = state_data["time_now"]
    elapsed_time = now - state_data["state_enter_time"]

    peg_tip = data.site("peg_tip").xpos.copy()
    prev_peg_tip = state_data["prev_peg_tip"].copy()

    hole_preinsert = state_data["hole_preinsert_world"]
    hole_entry = state_data["hole_entry_world"]
    hole_bottom = state_data["hole_bottom_world"]

    dt = max(1e-6, now - state_data["prev_time"])
    tip_vel = compute_velocity(peg_tip, prev_peg_tip, dt)

    tip_vz_down = -tip_vel[2]   # 아래 방향이면 +
    tip_vz_abs = abs(tip_vel[2])
    tip_xy_err_entry = np.linalg.norm((peg_tip - hole_entry)[:2])

    hole_top_z = hole_bottom[2] + state_data["hole_top_height_m"]
    near_hole_top = abs(peg_tip[2] - hole_top_z) <= state_data["contact_stall_top_band_m"]
    below_hole_top_10mm = peg_tip[2] <= (hole_top_z - state_data["contact_below_top_check_m"])

    if state == STATE_MOVE_PREINSERT:
        err = np.linalg.norm(peg_tip - hole_preinsert)
        if err < state_data["preinsert_tol"]:
            print("[STATE] MOVE_PREINSERT -> CONTACT_APPROACH")
            set_state(STATE_CONTACT_APPROACH, state_data, data, now)

    elif state == STATE_CONTACT_APPROACH:
        if near_hole_top and tip_vz_abs < state_data["contact_stall_v_thresh_mps"]:
            state_data["contact_stall_z"] = peg_tip[2]
            print(f"[STATE] CONTACT_APPROACH -> HOLE_SEARCH (stall_z={state_data['contact_stall_z']:.6f})")
            set_state(STATE_HOLE_SEARCH, state_data, data, now)

        elif below_hole_top_10mm:
            if tip_xy_err_entry > state_data["contact_retreat_xy_thresh"]:
                print("[STATE] CONTACT_APPROACH -> RETREAT (xy too far below hole top)")
                set_state(STATE_RETREAT, state_data, data, now)
            else:
                print("[STATE] CONTACT_APPROACH -> INSERT_FINAL")
                set_state(STATE_INSERT_FINAL, state_data, data, now)

    elif state == STATE_HOLE_SEARCH:
        freq = state_data["hole_search_wiggle_freq"]
        one_cycle_time = (1.0 / freq) if freq > 0.0 else 0.0
        decay_end_time = one_cycle_time + state_data["hole_search_timeout"]

        if elapsed_time >= decay_end_time:
            contact_stall_z = state_data["contact_stall_z"]

            if contact_stall_z is None:
                print("[STATE] HOLE_SEARCH -> RETREAT (contact_stall_z is None)")
                set_state(STATE_RETREAT, state_data, data, now)
            else:
                lowered_enough = peg_tip[2] < (contact_stall_z - state_data["hole_search_success_z_margin"])

                if lowered_enough:
                    print(
                        "[STATE] HOLE_SEARCH -> INSERT_FINAL "
                        f"(stall_z={contact_stall_z:.6f}, current_z={peg_tip[2]:.6f})"
                    )
                    set_state(STATE_INSERT_FINAL, state_data, data, now)
                else:
                    print(
                        "[STATE] HOLE_SEARCH -> RETREAT "
                        f"(stall_z={contact_stall_z:.6f}, current_z={peg_tip[2]:.6f})"
                    )
                    set_state(STATE_RETREAT, state_data, data, now)

    elif state == STATE_INSERT_FINAL:
        if peg_tip[2] <= state_data["insert_final_done_z_m"]:
            print("[STATE] INSERT_FINAL -> DONE")
            set_state(STATE_DONE, state_data, data, now)

        else:
            monitor_zone = peg_tip[2] > state_data["insert_final_stall_monitor_z_m"]
            stalled = tip_vz_abs < state_data["insert_final_stall_v_thresh_mps"]

            if monitor_zone and stalled:
                if state_data["insert_final_stall_start_time"] is None:
                    state_data["insert_final_stall_start_time"] = now
                elif (now - state_data["insert_final_stall_start_time"]) > state_data["insert_final_stall_hold_time"]:
                    print("[STATE] INSERT_FINAL -> RETREAT (timeout)")
                    set_state(STATE_RETREAT, state_data, data, now)
            else:
                state_data["insert_final_stall_start_time"] = None

    elif state == STATE_RETREAT:
        err = np.linalg.norm(peg_tip - state_data["retreat_target"])
        if err < state_data["retreat_tol"]:
            print("[STATE] RETREAT -> MOVE_PREINSERT")
            set_state(STATE_MOVE_PREINSERT, state_data, data, now)

    state_data["prev_peg_tip"] = peg_tip.copy()


def print_debug(data, state_data, desired_xpos_site, control_site_name, print_every=0.1):
    now = state_data["time_now"]
    if now - state_data["last_print_time"] < print_every:
        return

    state_data["last_print_time"] = now

    current_pos = data.site(control_site_name).xpos.copy()
    pos_err = current_pos - desired_xpos_site
    tau_norm = np.linalg.norm(state_data["last_tau_cmd"])
    bias_xy = state_data["target_xy_bias"][:2] * 1000.0

    print(
        f"[{get_state_name(state_data['state'])} | site={control_site_name}] "
        f"POS=({current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}) | "
        f"ERR=({pos_err[0]:.4f}, {pos_err[1]:.4f}, {pos_err[2]:.4f}) | "
        f"bias_xy_mm=({bias_xy[0]:.2f}, {bias_xy[1]:.2f}) | "
        f"tau={tau_norm:.3f}"
    )

def build_state_data(initial_peg_tip, hole_preinsert_world, hole_entry_world, hole_bottom_world, target_xy_bias):
    return {
        "state": STATE_MOVE_PREINSERT,
        "state_enter_time": 0.0,
        "time_now": 0.0,
        "last_print_time": -1.0,
        "prev_time": 0.0,

        "hole_preinsert_world": hole_preinsert_world,
        "hole_entry_world": hole_entry_world,
        "hole_bottom_world": hole_bottom_world,
        "target_xy_bias": target_xy_bias.copy(),

        "retreat_target": initial_peg_tip.copy(),
        "done_hold_target": initial_peg_tip.copy(),

        "preinsert_tol": 0.004,
        "touch_down_speed": 0.010,

        "hole_search_down_speed": 0.0005,
        "hole_search_wiggle_roll_deg": 10.0,
        "hole_search_wiggle_pitch_deg": 10.0,
        "hole_search_wiggle_freq": 0.3,
        "hole_search_xy_amp": 0.0,
        "hole_search_timeout": 1.0,

        "hole_found_start_time": None,
        "hole_found_vz_thresh": 0.0015,
        "hole_found_xy_thresh": 0.0010,
        "hole_found_z_margin": 0.0005,
        "hole_found_hold_time": 0.08,

        "hole_search_verify_duration": 3.0,
        "hole_search_verify_min_drop": 0.0004,
        "hole_search_verify_vz_thresh": 0.0005,

        "hole_search_verify_start_time": None,
        "hole_search_verify_start_z": None,

        "retreat_tol": 0.005,

        "phase_start_pos": initial_peg_tip.copy(),
        "prev_peg_tip": initial_peg_tip.copy(),
        "last_tau_cmd": np.zeros(6),

        "hole_top_height_m": HOLE_TOP_HEIGHT_M,
        "contact_stall_top_band_m": CONTACT_STALL_TOP_BAND_M,
        "contact_stall_v_thresh_mps": CONTACT_STALL_V_THRESH_MPS,
        "contact_below_top_check_m": CONTACT_BELOW_TOP_CHECK_M,
        "contact_retreat_xy_thresh": CONTACT_RETREAT_XY_THRESH_M,
        "contact_push_below_top_m": CONTACT_PUSH_BELOW_TOP_M,
        "retreat_lift_m": RETREAT_LIFT_M,

        "contact_stall_z": None,
        "hole_search_success_z_margin": 0.0001,

        "insert_final_down_speed": 0.010,
        "insert_final_target_z_m": 0.005,
        "insert_final_stall_monitor_z_m": 0.008,
        "insert_final_stall_v_thresh_mps": 0.0005,
        "insert_final_stall_hold_time": 2.0,
        "insert_final_stall_start_time": None,
        "insert_final_done_z_m": 0.006,
    }


def reset_trial(model, data, rng):
    mujoco.mj_resetData(model, data)
    initialize_robot_state(model, data)
    mujoco.mj_forward(model, data)

    hole_preinsert_world, hole_entry_world, hole_bottom_world, target_xy_bias = build_biased_hole_targets(data, rng)
    initial_peg_tip = data.site("peg_tip").xpos.copy()

    print(
        f"[TARGET_BIAS] dx={target_xy_bias[0] * 1000:.2f} mm, "
        f"dy={target_xy_bias[1] * 1000:.2f} mm, "
        f"r={np.linalg.norm(target_xy_bias[:2]) * 1000:.2f} mm"
    )

    state_data = build_state_data(
        initial_peg_tip,
        hole_preinsert_world,
        hole_entry_world,
        hole_bottom_world,
        target_xy_bias,
    )

    return state_data


def resolve_model_path():
    local_path = Path(__file__).resolve().parent / "scene_rb5_peg_in_hole.xml"
    if local_path.exists():
        return str(local_path)
    return "/home/chu/manipulator_control/rb5/peg_in_hole/scene_rb5_peg_in_hole.xml"


def main():
    model_path = resolve_model_path()
    model, data = create_model_and_data(model_path)

    validate_required_sites(model)
    initialize_robot_state(model, data)

    mass, gravity, jacp, jacr, c_joint = create_work_buffers(model)

    k_pos = np.array([60.0, 60.0, 35.0])
    zeta_pos = np.array([8.0, 8.0, 4.0])

    k_ori = np.array([3.0, 3.0, 1.5])
    zeta_ori = np.array([0.3, 0.3, 0.2])

    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(RANDOM_SEED)

    trial_idx = 1
    state_data = reset_trial(model, data, rng)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if VIEW_HOLE:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            hole_view = data.site("hole_entry").xpos.copy()
            viewer.cam.lookat[:] = hole_view
            viewer.cam.distance = 0.30
            viewer.cam.azimuth = 160.0
            viewer.cam.elevation = -20.0

        print(f"\n========== TRIAL {trial_idx}/{TOTAL_TRIALS} ==========\n")
        t0 = time()

        while viewer.is_running():
            now = time() - t0
            state_data["time_now"] = now

            desired_xpos_site, desired_rpy, control_site_name = get_target_by_state(state_data)

            compute_mass_and_gravity(model, data, mass, gravity)

            if state_data["state"] in (STATE_HOLE_SEARCH, STATE_INSERT_FINAL):
                torque = compute_task_torque(
                    model,
                    data,
                    mass,
                    gravity,
                    jacp,
                    jacr,
                    c_joint,
                    desired_xpos_site,
                    desired_rpy,
                    k_pos,
                    zeta_pos,
                    k_ori,
                    zeta_ori,
                    site_name=control_site_name,
                    pos_task_axis_mask=np.array([0.0, 0.0, 1.0]),
                )
            else:
                torque = compute_task_torque(
                    model,
                    data,
                    mass,
                    gravity,
                    jacp,
                    jacr,
                    c_joint,
                    desired_xpos_site,
                    desired_rpy,
                    k_pos,
                    zeta_pos,
                    k_ori,
                    zeta_ori,
                    site_name=control_site_name,
                )

            apply_control(data, torque, max_torque=80)
            state_data["last_tau_cmd"] = data.ctrl[0:6].copy()

            prev_state = state_data["state"]

            mujoco.mj_step(model, data)

            print_debug(data, state_data, desired_xpos_site, control_site_name, print_every=0.1)
            update_state(data, state_data)

            curr_state = state_data["state"]

            # trial 종료 조건
            trial_finished = False

            # 성공: DONE 도달
            if curr_state == STATE_DONE:
                trial_finished = True

            # 실패: RETREAT가 끝나서 다시 MOVE_PREINSERT로 복귀
            elif prev_state == STATE_RETREAT and curr_state == STATE_MOVE_PREINSERT:
                trial_finished = True

            if trial_finished:
                if trial_idx >= TOTAL_TRIALS:
                    print("\n========== ALL TRIALS FINISHED ==========\n")
                    viewer.sync()
                    break

                trial_idx += 1
                print(f"\n========== TRIAL {trial_idx}/{TOTAL_TRIALS} ==========\n")

                state_data = reset_trial(model, data, rng)
                t0 = time()

                viewer.sync()
                continue

            viewer.sync()
            state_data["prev_time"] = now


if __name__ == "__main__":
    main()