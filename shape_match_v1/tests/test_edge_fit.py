"""Offline audit checks; run with numpy/scipy/Pillow installed."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_edge_fit import analyze, transform

data, prediction, reference = map(Path, sys.argv[1:])
p = np.array([[0., 0.], [3., -4.]])
pose = dict(angle_deg=0, scale_row=1, row=11, column=23)
np.testing.assert_allclose(transform(p, pose), p+[23, 11])
pose.update(angle_deg=90, scale_row=2)
np.testing.assert_allclose(transform(p, pose),
                           np.column_stack((2*(p[:, 1]+.5)+22.5, -2*(p[:, 0]+.5)+10.5)))
forward = analyze(data, prediction, reference)
reverse = analyze(data, reference, prediction)
identical = analyze(data, prediction, prediction)
for a, b, same in zip(forward["experiments"], reverse["experiments"], identical["experiments"]):
    assert a["valid"] == b["valid"]
    for key in ("rmse_px", "mean_signed_px", "p95_abs_px"):
        np.testing.assert_allclose(a["halcon"][key], b["reproduction"][key], atol=1e-12)
        np.testing.assert_allclose(a["reproduction"][key], b["halcon"][key], atol=1e-12)
        np.testing.assert_allclose(same["halcon"][key], same["reproduction"][key], atol=1e-12)
print("PASS: XLD coordinate conversion, role-swap symmetry, identical-pose equality")
