import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from calibration.country_wise_calibration.bootstrap_globals import OUTPUT_PATH

with open(OUTPUT_PATH) as f:
    pack = json.load(f)
required = {"kappa", "sigma", "beta_trade_off", "h_P", "s_pi", "s_pi_P"}
assert required <= pack.keys(), f"missing: {required - pack.keys()}"
for k in required:
    assert isinstance(pack[k], float), f"{k} is not float"
print("ok bootstrap pack")
