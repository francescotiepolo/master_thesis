"""
Stage 4: free all-country simulation over calibration (1988-2014) and
validation (2014-2024) windows, plus losses and plots.
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from calibration.calibration_config import CALIB_DIR, YEAR_START, YEAR_END
from calibration.calibration_utils import aggregate_loss_components, load_data, spearman
from calibration.country_wise_calibration.calibration_country_wise import (
    COUNTRY_CALIB_END, COUNTRY_CALIB_YEARS, COUNTRY_VALID_YEARS,
    load_country_index, load_growth_regression_r,
)
from calibration.country_wise_calibration.shared_global_calibration import (
    GLOBAL_PARAM_NAMES, build_joint_model, free_simulate_all_countries,
)
from calibration.country_wise_calibration.joint_params import JOINT_JSON, JOINT_NPZ

OUTPUT_DIR = os.path.join(CALIB_DIR, "joint")
VALIDATION_MODE = os.environ.get("JOINT_VALIDATION_MODE", "continuous").strip().lower()


_REQUIRED_NPZ_KEYS = (
    "s_pi", "G", "nu", "h_C", "entry_threshold", "beta_trade_off",
)


def _load_joint_pack():
    """
    Load the canonical joint parameter pack produced by joint_params.py (Stage 4 prep).
    Raises FileNotFoundError if the pack is missing, KeyError if its schema is
    stale relative to the current GLOBAL_PARAM_NAMES / per-country layout.
    """
    if not os.path.exists(JOINT_JSON):
        raise FileNotFoundError(
            f"Missing joint pack: {JOINT_JSON}. "
            "Run calibration/country_wise_calibration/joint_params.py first."
        )
    if not os.path.exists(JOINT_NPZ):
        raise FileNotFoundError(
            f"Missing joint pack arrays: {JOINT_NPZ}. "
            "Run calibration/country_wise_calibration/joint_params.py first."
        )
    with open(JOINT_JSON) as f:
        meta = json.load(f)
    arr = np.load(JOINT_NPZ)

    globals_pack = meta.get("globals", {})
    missing_globals = [k for k in GLOBAL_PARAM_NAMES if k not in globals_pack]
    if missing_globals:
        raise KeyError(
            f"{JOINT_JSON} is missing required global keys {missing_globals}. "
            f"The pack is stale relative to GLOBAL_PARAM_NAMES = {GLOBAL_PARAM_NAMES}. "
            "Re-run Stage 3 then joint_params.py."
        )
    missing_arr = [k for k in _REQUIRED_NPZ_KEYS if k not in arr.files]
    if missing_arr:
        raise KeyError(
            f"{JOINT_NPZ} is missing required arrays {missing_arr}. "
            "The pack is stale relative to the current per-country schema. "
            "Re-run joint_params.py."
        )

    global_theta = [float(globals_pack[name]) for name in GLOBAL_PARAM_NAMES]
    country_vecs = {
        "s_pi":            arr["s_pi"],
        "G":               arr["G"],
        "nu":              arr["nu"],
        "h_C":             arr["h_C"],
        "entry_threshold": arr["entry_threshold"],
        "beta_trade_off":  arr["beta_trade_off"],
    }
    if "beta_trade_off_corrected" in arr.files:
        country_vecs["beta_trade_off_corrected"] = arr["beta_trade_off_corrected"]
    return global_theta, country_vecs


def _save_traj(path, traj):
    years = sorted(traj.keys())
    SP = traj[years[0]][2].shape[0]
    SC = traj[years[0]][1].shape[0]
    A = np.stack([traj[y][0] for y in years])  # (T, SC, SP)
    C = np.stack([traj[y][1] for y in years])  # (T, SC)
    P = np.stack([traj[y][2] for y in years])  # (T, SP)
    np.savez(path, years=np.array(years), alpha=A, C=C, P=P)


def run_calibration_window():
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    global_theta, country_vecs = _load_joint_pack()
    model = build_joint_model(global_theta, data, country_vecs)
    traj = free_simulate_all_countries(model, data, COUNTRY_CALIB_YEARS, start_year=YEAR_START)
    if traj is None:
        raise RuntimeError("calibration-window free run failed")
    _save_traj(os.path.join(OUTPUT_DIR, "trajectory_calib.npz"), traj)
    return traj, data, rows


def run_validation_window(traj_calib=None):
    data = load_data()
    rows = load_country_index()
    data["r_C_growth_regression"] = load_growth_regression_r(rows)
    global_theta, country_vecs = _load_joint_pack()

    # Validation default is a continuous rollout from the simulated 2014 state.
    # Set JOINT_VALIDATION_MODE=reseed to run the legacy observed-state restart.
    start_year = COUNTRY_CALIB_END
    data_seeded = dict(data)
    if VALIDATION_MODE == "reseed":
        data_seeded["alpha_init"] = data["alpha_obs"][start_year].copy()
        data_seeded["C_init"] = data["C_obs"][start_year].copy()
        data_seeded["P_init"] = data["P_obs"][start_year].copy()
        print("Validation mode: reseed from observed 2014 state")
    else:
        if traj_calib is None or start_year not in traj_calib:
            raise RuntimeError(
                "Validation mode 'continuous' requires calibration trajectory "
                "with the 2014 state."
            )
        alpha_2014, C_2014, P_2014 = traj_calib[start_year]
        data_seeded["alpha_init"] = alpha_2014.copy()
        data_seeded["C_init"] = C_2014.copy()
        data_seeded["P_init"] = P_2014.copy()
        print("Validation mode: continuous from simulated 2014 state")

    model = build_joint_model(global_theta, data_seeded, country_vecs)
    # model.alpha is already set from data_seeded["alpha_init"] inside
    # _patch_model (calibration_utils.py line 274), so no explicit assignment needed.

    traj = free_simulate_all_countries(
        model, data_seeded, COUNTRY_VALID_YEARS, start_year=start_year,
    )
    if traj is None:
        raise RuntimeError("validation-window free run failed")
    _save_traj(os.path.join(OUTPUT_DIR, "trajectory_valid.npz"), traj)
    return traj, data, rows


def compute_and_save_losses(traj_calib, traj_valid, data):
    out = {
        "calibration": aggregate_loss_components(traj_calib, data, years=COUNTRY_CALIB_YEARS),
        "validation":  aggregate_loss_components(traj_valid, data, years=COUNTRY_VALID_YEARS),
    }
    with open(os.path.join(OUTPUT_DIR, "loss_components.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("loss components:")
    print(json.dumps(out, indent=2))


def _country_share_series(traj, data, country_idx):
    years = sorted(traj.keys())
    sim = []
    obs = []
    for y in years:
        _, C_sim, _ = traj[y]
        sim.append(float(C_sim[country_idx] / C_sim.sum()))
        obs.append(float(data["C_obs"][y][country_idx] / data["C_obs"][y].sum()))
    return years, sim, obs


def plot_trajectory_fit(traj_calib, traj_valid, data, rows):
    n_cols, n_rows = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)
    for idx, r in enumerate(rows):
        ax = axes.flat[idx]
        ci = int(r["position"])
        yc, sc, oc = _country_share_series(traj_calib, data, ci)
        yv, sv, ov = _country_share_series(traj_valid, data, ci)
        ax.plot(yc + yv, oc + ov, color="#1F4E79", lw=1.5, label="Observed")
        if yc:
            ax.plot(yc, sc, color="#B5541E", lw=1.5, label="Simulated")
        if yv:
            ax.plot(yv, sv, color="#B5541E", lw=1.5, ls="--")
        ax.axvline(COUNTRY_CALIB_END, color="grey", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(r["country_name_short"], fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    for idx in range(len(rows), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Year"); fig.supylabel("Market share")
    fig.suptitle("Joint free all-country simulation: observed vs simulated",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_fit.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_trajectory_summary(traj_calib, traj_valid, data, rows):
    yc = sorted(traj_calib.keys()); yv = sorted(traj_valid.keys())
    years = yc + yv
    if not years:
        return
    SC = len(rows)
    obs_mat = np.full((SC, len(years)), np.nan)
    sim_mat = np.full((SC, len(years)), np.nan)
    year_to_col = {y: k for k, y in enumerate(years)}
    for r in rows:
        ci = int(r["position"])
        for traj in (traj_calib, traj_valid):
            for y in sorted(traj.keys()):
                _, C_sim, _ = traj[y]
                obs_mat[ci, year_to_col[y]] = data["C_obs"][y][ci] / data["C_obs"][y].sum()
                sim_mat[ci, year_to_col[y]] = C_sim[ci] / C_sim.sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    yarr = np.asarray(years)
    for i in range(SC):
        ax.plot(yarr, obs_mat[i], color="#4C72B0", alpha=0.08, lw=0.7)
        ax.plot(yarr, sim_mat[i], color="#DD8452", alpha=0.08, lw=0.7)
    ax.plot(yarr, np.nanmedian(obs_mat, axis=0), color="#1F4E79", lw=2.5, label="Observed median")
    ax.plot(yarr, np.nanmedian(sim_mat, axis=0), color="#B5541E", lw=2.5, label="Simulated median")
    ax.axvline(COUNTRY_CALIB_END, color="grey", ls=":", lw=1, alpha=0.7, label="Calibration boundary")
    ax.set_xlabel("Year"); ax.set_ylabel("Market share")
    ax.set_title("Joint free all-country simulation summary", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    traj_calib, data, rows = run_calibration_window()
    traj_valid, _, _ = run_validation_window(traj_calib=traj_calib)
    compute_and_save_losses(traj_calib, traj_valid, data)
    plot_trajectory_fit(traj_calib, traj_valid, data, rows)
    plot_trajectory_summary(traj_calib, traj_valid, data, rows)
    print(f"Wrote artefacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
