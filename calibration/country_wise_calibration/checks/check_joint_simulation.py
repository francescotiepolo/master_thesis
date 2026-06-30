import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["STAGE3_COUNTRY_SUMMARY"] = "calibration/calibration_results/country_wise/summary.csv"
from calibration.country_wise_calibration.joint_params import build_joint_param_pack
from calibration.country_wise_calibration.joint_simulation import run_validation_window

if __name__ == "__main__":
    build_joint_param_pack()
    traj, data, rows = run_validation_window()
    assert len(traj) > 0
    print("ok joint validation smoke, years simulated:", sorted(traj.keys()))
