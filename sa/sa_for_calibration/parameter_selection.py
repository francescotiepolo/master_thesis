import os
import csv
import json
import numpy as np

S2_ROBUST_MIN = 0.03
S2_MAX_RESCUED = None  # None = all candidates above S2_ROBUST_MIN are rescued
TARGET_FREE_PARAMS = 7
SELECTION_OUTPUT_WEIGHTS = {
    "loss_total": 0.40,
    "nrmse_C": 0.20,
    "traj_corr_C": 0.20,
    "rank_products": 0.10,
    "nrmse_P": 0.10,
}


def recommend_free_fixed(result, problem_names, output_names, results_dir,
                         selection_output_weights, target_free_params=6):
    """
    Recommend free vs fixed parameters for calibration from Sobol indices.

    Robust (takes uncertainty into account) score per parameter:
      sum_w max(ST - ST_conf, 0)
    and:
      sum_w ST

    Second-order integration:
      - Compute weighted robust pair interaction:
          robust_s2(i,j) = sum_w max(S2(i,j) - S2_conf(i,j), 0)
      - If robust_s2 >= S2_ROBUST_MIN and one member is already free,
        the other parameter is an interaction candidate.
      - Rescue up to S2_MAX_RESCUED candidates with highest robust_s2.
    """
    rows = []

    for j, pname in enumerate(problem_names):
        st = {out: float(result["Si"][out]["ST"][j]) for out in output_names}
        stc = {out: float(result["Si"][out]["ST_conf"][j]) for out in output_names}
        lcb = {out: max(st[out] - stc[out], 0.0) for out in output_names}

        robust_score = sum(selection_output_weights[out] * lcb[out] for out in output_names)
        score = sum(selection_output_weights[out] * st[out] for out in output_names)

        # Coefficient of variation: high value means sensitivity is concentrated in one output
        st_vals = np.array([st[out] for out in output_names])
        st_mean = st_vals.mean()
        output_cv = float(st_vals.std() / st_mean) if st_mean > 1e-9 else 0.0

        row = {
            "parameter": pname,
            "robust_score": robust_score,
            "score": score,
            "output_cv": output_cv,
        }
        for out in output_names:
            row[f"ST_{out}"] = st[out]
            row[f"STconf_{out}"] = stc[out]
        rows.append(row)

    rows.sort(key=lambda r: (r["robust_score"], r["score"]), reverse=True)
    free_base = [r["parameter"] for r in rows[:target_free_params]]
    free_set = set(free_base)

    # Warn if the boundary between free and fixed is small (gap < 20% of last-selected score)
    if len(rows) > target_free_params:
        last_in = rows[target_free_params - 1]["robust_score"]
        first_out = rows[target_free_params]["robust_score"]
        if last_in > 0 and (last_in - first_out) / last_in < 0.20:
            import warnings
            warnings.warn(
                f"Narrow free/fixed boundary: "
                f"'{rows[target_free_params - 1]['parameter']}' (robust={last_in:.4f}) vs "
                f"'{rows[target_free_params]['parameter']}' (robust={first_out:.4f}). "
                f"Selection may be sensitive to target_free_params or SA sample size.",
                stacklevel=2,
            )

    # Build weighted robust S2 for each parameter pair
    robust_s2_pairs = []
    for i in range(len(problem_names)):
        for j in range(i + 1, len(problem_names)):
            pi = problem_names[i]
            pj = problem_names[j]
            robust_ij = 0.0
            s2_by_output = {}
            for out in output_names:
                s2_mat = result["Si"][out].get("S2")
                s2c_mat = result["Si"][out].get("S2_conf")
                if s2_mat is None or s2c_mat is None:
                    continue
                s2_val = float(s2_mat[i][j])
                s2_conf = float(s2c_mat[i][j])
                robust_ij += selection_output_weights[out] * max(s2_val - s2_conf, 0.0)
                s2_by_output[out] = s2_val
            robust_s2_pairs.append((robust_ij, pi, pj, s2_by_output))

    # Identify candidates for rescue based on robust S2 interactions with already free parameters
    rescued = []
    if robust_s2_pairs:
        robust_s2_pairs.sort(key=lambda x: x[0], reverse=True)
        candidate_strength = {}
        for robust_ij, pi, pj, _ in robust_s2_pairs:
            if robust_ij < S2_ROBUST_MIN:
                break
            pi_free = pi in free_set
            pj_free = pj in free_set
            if pi_free ^ pj_free:
                cand = pj if pi_free else pi
                if cand not in free_set:
                    # Accumulate over all qualifying pairs so a parameter that
                    # interacts moderately with many free params scores higher
                    # than one that interacts strongly with just one
                    candidate_strength[cand] = candidate_strength.get(cand, 0.0) + robust_ij

        sorted_candidates = sorted(candidate_strength.keys(),
                                   key=lambda p: candidate_strength[p],
                                   reverse=True)
        rescued = sorted_candidates if S2_MAX_RESCUED is None else sorted_candidates[:S2_MAX_RESCUED]

    free = free_base + rescued
    fixed = [r["parameter"] for r in rows if r["parameter"] not in set(free)]

    scores_path = os.path.join(results_dir, "parameter_selection_scores.csv")
    st_cols = [f"ST_{out}"     for out in output_names]
    stc_cols = [f"STconf_{out}" for out in output_names]
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "parameter", "robust_score", "score", "output_cv"]
            + st_cols + stc_cols
            + ["recommended_status", "selection_reason"]
        )
        for i, r in enumerate(rows, start=1):
            pname = r["parameter"]
            if pname in free_base:
                reason = "top_ST"
            elif pname in rescued:
                reason = "rescued_by_S2"
            else:
                reason = "not_selected"
            writer.writerow(
                [i, r["parameter"], round(r["robust_score"], 6), round(r["score"], 6),
                 round(r["output_cv"], 4)]
                + [round(r[c], 6) for c in st_cols]
                + [round(r[c], 6) for c in stc_cols]
                + ["free" if pname in free else "fixed", reason]
            )

    rec_path = os.path.join(results_dir, "parameter_selection_recommendation.json")
    payload = {
        "criterion": {
            "target_free_params": int(target_free_params),
            "selection_output_weights": selection_output_weights,
            "definition": "rank by weighted max(ST-ST_conf,0); tie-break by weighted ST; then S2 rescue",
            "s2_robust_min": S2_ROBUST_MIN,
            "s2_max_rescued": S2_MAX_RESCUED,
        },
        "recommended_free_base": free_base,
        "recommended_free_rescued_by_s2": rescued,
        "recommended_free": free,
        "recommended_fixed": fixed,
    }
    with open(rec_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nRecommended free/fixed parameters:")
    print("  Free:", ", ".join(free))
    print("  Fixed:", ", ".join(fixed))
    print("\nParameter ranking (magnitude):")
    st_header = "  ".join(f"ST_{o[:6]}" for o in output_names)
    print(f"  rank  parameter          robust    score     {st_header}  cv    status")
    for i, r in enumerate(rows, start=1):
        pname = r["parameter"]
        status = "free" if pname in free else "fixed"
        het_flag = " (het)" if r["output_cv"] > 0.8 else ""
        st_vals = "   ".join(f"{r[f'ST_{o}']:.4f}" for o in output_names)
        print(
            f"  {i:>2d}    "
            f"{pname:<16} "
            f"{r['robust_score']:.4f}   "
            f"{r['score']:.4f}   "
            f"{st_vals}   "
            f"{r['output_cv']:.2f}  "
            f"{status}{het_flag}"
        )
    # Print top second-order interactions
    if robust_s2_pairs and any(v > 0 for v, _, _, _ in robust_s2_pairs):
        top_pairs = sorted(robust_s2_pairs, key=lambda x: x[0], reverse=True)[:8]
        print("\nTop second-order interactions (robust):")
        for robust_ij, pi, pj, s2_by_output in top_pairs:
            if robust_ij <= 0:
                continue
            pair = f"{pi} x {pj}"
            s2_str = "  ".join(f"{s2_by_output.get(o, 0.0):.4f}" for o in output_names)
            print(f"  {pair:<28} robust={robust_ij:.4f}  S2: {s2_str}")
    print(f"Saved: {scores_path}")
    print(f"Saved: {rec_path}")

    return {"free": free, "fixed": fixed, "rows": rows}


def _load_result_from_sobol_csv(csv_path, s2_csv_path=None):
    """
    Reconstruct necessary Sobol result dict from sobol_indices.csv.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    output_names = []
    problem_names = []
    for r in rows:
        out = r["output"]
        pname = r["parameter"]
        if out not in output_names:
            output_names.append(out)
        if pname not in problem_names:
            problem_names.append(pname)

    si = {}
    for out in output_names:
        out_rows = [r for r in rows if r["output"] == out]
        out_map = {r["parameter"]: r for r in out_rows}
        si[out] = {
            "ST": np.array([float(out_map[p]["ST"]) for p in problem_names]),
            "ST_conf": np.array([float(out_map[p]["ST_conf"]) for p in problem_names]),
        }

    if s2_csv_path and os.path.exists(s2_csv_path):
        s2_rows = []
        with open(s2_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s2_rows.append(row)

        idx = {p: i for i, p in enumerate(problem_names)}
        n = len(problem_names)
        for out in output_names:
            mat = np.zeros((n, n))
            matc = np.zeros((n, n))
            for r in s2_rows:
                if r["output"] != out:
                    continue
                i = idx[r["param_i"]]
                j = idx[r["param_j"]]
                s2v = float(r["S2"])
                s2cv = float(r["S2_conf"])
                mat[i, j] = s2v
                mat[j, i] = s2v
                matc[i, j] = s2cv
                matc[j, i] = s2cv
            si[out]["S2"] = mat
            si[out]["S2_conf"] = matc

    return {"Si": si}, problem_names, output_names


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    sobol_csv_path = os.path.join(results_dir, "sobol_indices.csv")
    sobol_s2_csv_path = os.path.join(results_dir, "sobol_indices_s2.csv")
    result, problem_names, output_names = _load_result_from_sobol_csv(
        sobol_csv_path, s2_csv_path=sobol_s2_csv_path
    )

    recommend_free_fixed(
        result=result,
        problem_names=problem_names,
        output_names=output_names,
        results_dir=results_dir,
        selection_output_weights=SELECTION_OUTPUT_WEIGHTS,
        target_free_params=TARGET_FREE_PARAMS,
    )