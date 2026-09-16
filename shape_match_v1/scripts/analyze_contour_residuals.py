#!/usr/bin/env python3
"""Offline frozen-correspondence diagnosis. Requires numpy; never changes matching."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from compare_gt import compare, read_rows


def analyze(directory, reference):
    predictions = read_rows(directory / "baseline.csv")
    references = read_rows(reference)
    association = compare(predictions, references)
    if association["associated"] != len(references) or association["extra"]:
        raise ValueError("Complete unambiguous baseline association required")
    by_ref = {r["index"]: r for r in references}
    ref_for_prediction = {r["prediction_index"]: by_ref[r["reference_index"]]
                          for r in association["comparisons"]}
    with (directory / "contour_residuals.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    reassociated = defaultdict(list)
    path = directory / "contour_residuals_at_halcon.csv"
    if path.exists():
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                reassociated[int(row["prediction_index"]), row["method"], row["component"]].append(row)
    instances = defaultdict(list)
    for row in rows:
        instances[int(row["prediction_index"]), row["method"]].append(row)
    output = []
    for (index, method), instance in sorted(instances.items()):
        ref = ref_for_prediction[index]
        theta = -math.radians(ref["angle_deg"])
        scale = ref["scale_row"]
        c, s = scale * math.cos(theta), scale * math.sin(theta)
        # Invert the same edge-centered result-coordinate convention as result.cpp.
        tx = ref["column"] - .5 + .5 * (c - s)
        ty = ref["row"] - .5 + .5 * (c + s)
        rotation = np.array([[c, -s], [s, c]])

        def matrices(group):
            valid = [r for r in group if r["valid"] == "1"]
            j = np.array([[float(r[k]) for k in ("j_tx", "j_ty", "j_theta", "j_logscale")]
                          for r in valid]).reshape(-1, 4)
            residual = np.array([float(r["residual"]) for r in valid])
            p = np.array([[float(r[k]) for k in ("model_x", "model_y")] for r in valid]).reshape(-1, 2)
            q = np.array([[float(r[k]) for k in ("scene_x", "scene_y")] for r in valid]).reshape(-1, 2)
            frozen_gt_residual = np.sum((p @ rotation.T + [tx, ty] - q) * j[:, :2], axis=1)
            return j, residual, frozen_gt_residual

        j_all, _, _ = matrices(instance)
        joint_inverse = np.linalg.pinv(j_all.T @ j_all, rcond=1e-12)
        groups = defaultdict(list)
        for row in instance:
            groups[row["component"]].append(row)
        for component, group in groups.items():
            j, residual, gt_residual = matrices(group)
            h = j.T @ j
            diagonal = np.sqrt(np.maximum(np.diag(h), 1e-30))
            eigenvalues = np.linalg.eigvalsh(h / np.outer(diagonal, diagonal))
            condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None
            contribution = -joint_inverse @ j.T @ residual
            isolated = -np.linalg.pinv(h, rcond=1e-12) @ j.T @ residual
            units = np.array([1, 1, 180 / math.pi, 1])
            observed = reassociated[index, method, component]
            gt_values = np.array([float(r["residual"]) for r in observed if r["valid"] == "1"])
            radius = 1.5 if method == "nearest_1.5" else 2
            support_loss = lambda values, total: float(np.minimum(radius**2, values**2).sum() + (total-len(values))*radius**2)
            output.append({"reference_index": ref["index"], "prediction_index": index,
                "model": ref["model"], "method": method, "component": int(component),
                "template_points": len(group), "valid_points": len(residual),
                "coverage": len(residual) / len(group),
                "model_span_xy": [float(np.ptp([float(r[k]) for r in group])) for k in ("model_x", "model_y")],
                "baseline_normal_rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
                "halcon_pose_frozen_pairs_rmse": float(np.sqrt(np.mean(gt_residual**2))) if len(residual) else None,
                "halcon_pose_reassociated_rmse": float(np.sqrt(np.mean(gt_values**2))) if len(gt_values) else None,
                "halcon_pose_reassociated_valid_points": len(gt_values) if observed else None,
                "baseline_support_loss": support_loss(residual, len(group)),
                "halcon_pose_reassociated_support_loss": support_loss(gt_values, len(observed)) if observed else None,
                "mean_signed_residual": float(np.mean(residual)) if len(residual) else None,
                "joint_step_contribution_tx_ty_theta_deg_logscale": (contribution * units).tolist(),
                "isolated_linear_step_tx_ty_theta_deg_logscale": (isolated * units).tolist(),
                "normalized_hessian_condition": condition})
    return {"notes": ["GT is used only for evaluating frozen correspondences, not for optimization.",
                       "Group steps are local linear diagnostics, not final pose estimates.",
                       "Frozen-pair GT loss is not the re-associated HALCON objective."], "groups": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, args.reference), indent=2, allow_nan=False))
