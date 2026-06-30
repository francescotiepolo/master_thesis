"""
Orchestrate stages 2-4 of the joint pipeline. Stage 1 (model refactor) is a
one-time code change and is not invoked from here.

Usage:
    python run_joint_pipeline.py --stage product_reg # Product growth regression
    python run_joint_pipeline.py --stage bootstrap   # Stage 2 prep
    python run_joint_pipeline.py --stage country     # Stage 2 run
    python run_joint_pipeline.py --stage globals     # Stage 3
    python run_joint_pipeline.py --stage pack        # Stage 4 prep
    python run_joint_pipeline.py --stage simulate    # Stage 4 run
    python run_joint_pipeline.py --stage all         # all of the above
"""
import argparse
import sys


STAGES = ["product_reg", "bootstrap", "country", "globals", "pack", "simulate"]


def run(stage):
    if stage == "product_reg":
        from calibration.product_growth_regression import main as pr_main
        pr_main()
    elif stage == "bootstrap":
        from calibration.country_wise_calibration.bootstrap_globals import build_bootstrap_pack
        build_bootstrap_pack()
    elif stage == "country":
        from calibration.country_wise_calibration.calibration_country_wise import main as cw_main
        cw_main()
    elif stage == "globals":
        from calibration.country_wise_calibration.shared_global_calibration import main as sg_main
        sg_main()
    elif stage == "pack":
        from calibration.country_wise_calibration.joint_params import build_joint_param_pack
        build_joint_param_pack()
    elif stage == "simulate":
        from calibration.country_wise_calibration.joint_simulation import main as js_main
        js_main()
    else:
        raise ValueError(f"unknown stage {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["all"], required=True)
    args = parser.parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"\n=== Stage: {s} ===\n")
        run(s)


if __name__ == "__main__":
    main()
