"""Verify vector and scalar inputs both work for G, nu, entry_threshold."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from calibration.calibration_utils import build_model, load_data, _patch_model

data = load_data()
theta = [0.1, 0.5, 1.0, 0.3, 1.0, 0.01, 0.5, 0.5, 0.0, 0.05, 0.05, 5.0]

m_scalar = build_model(theta, data)
print("scalar:", m_scalar.G.shape, m_scalar.nu.shape, m_scalar.entry_threshold.shape)
assert m_scalar.G.shape == (data["SC"],)
assert m_scalar.nu.shape == (data["SC"],)
assert m_scalar.entry_threshold.shape == (data["SC"],)
assert np.allclose(m_scalar.G, m_scalar.G[0])
print("ok scalar broadcasts to vector")

m = build_model(theta, data)
G_vec = np.linspace(0.1, 1.0, data["SC"])
_patch_model(m, data, params=None, G_vec=G_vec)
assert np.allclose(m.G, G_vec)
print("ok _patch_model vector override")
