#!/usr/bin/env python3
"""Independent, shared-observation edge-fit audit; never optimizes input poses.

Requires numpy, scipy and Pillow. Reports normal-distance fit, not ground-truth
pose accuracy. Both poses use the same HALCON template and scene observations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from compare_gt import compare, read_rows


def contours(path):
    groups = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            groups.setdefault(int(r["contour_index"]), []).append(
                [float(r["column"]), float(r["row"])])
    for key, values in groups.items():
        p = np.asarray(values)
        if np.linalg.norm(p[0] - p[-1]) < 1e-4:
            p = p[:-1]
        yield key, p


def transform(p, pose):
    a = np.deg2rad(-pose["angle_deg"])
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    # Exported contours and result translation use HALCON's XLD convention.
    return (p + .5) @ rotation.T * pose["scale_row"] + [pose["column"], pose["row"]] - .5


def stats(x):
    return {"rmse_px": float(np.sqrt(np.mean(x*x))),
            "mean_signed_px": float(np.mean(x)),
            "p95_abs_px": float(np.percentile(np.abs(x), 95))} if len(x) else None


def analyze(data, predictions, reference):
    pred, refs = read_rows(predictions), read_rows(reference)
    assoc = compare(pred, refs)
    if assoc["associated"] != len(refs) or assoc["extra"]:
        raise ValueError("Complete, unique instance association required")
    pred_by_index = {r["index"]: r for r in pred}
    paired = {r["reference_index"]: pred_by_index[r["prediction_index"]]
              for r in assoc["comparisons"]}
    image = np.asarray(Image.open(data / "original.pgm"), dtype=float)
    result = {"reference": str(reference), "predictions": str(predictions),
              "method": "Shared HALCON template; midpoint-anchored Gaussian derivative peaks; "
                        "same observations and valid support for both poses; normal distances only",
              "radius_px": 2.5, "sample_step_px": .05, "experiments": []}
    offsets = np.linspace(-2.5, 2.5, 101)
    for sigma in (.6, 1., 1.4):
        gx = gaussian_filter(image, sigma, order=(0, 1))
        gy = gaussian_filter(image, sigma, order=(1, 0))
        groups, all_h, all_p, total = [], [], [], 0
        for ref in refs:
            for component, model in contours(data / "reference" / f'template_{ref["model"]}_contours.csv'):
                h, p = transform(model, ref), transform(model, paired[ref["index"]])
                center = (h+p)/2
                tangent = np.roll(center, -2, axis=0) - np.roll(center, 2, axis=0)
                normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
                normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
                positions = center[:, None, :] + normal[:, None, :]*offsets[None, :, None]
                coords = [positions[:, :, 1], positions[:, :, 0]]
                dx, dy = [map_coordinates(g, coords, order=3, mode="nearest") for g in (gx, gy)]
                response = np.abs(dx*normal[:, 0, None]+dy*normal[:, 1, None])
                peaks = (response[:, 1:-1] > response[:, :-2]) & (response[:, 1:-1] >= response[:, 2:])
                peaks &= response[:, 1:-1] >= 5
                # Pick nearest shared-anchor peak, never the peak closest to either pose.
                distance = np.where(peaks, np.abs(offsets[1:-1])[None, :], np.inf)
                k = np.argmin(distance, axis=1)+1
                ii = np.arange(len(model))
                valid = np.isfinite(distance.min(axis=1))
                left, mid, right = response[ii, k-1], response[ii, k], response[ii, k+1]
                denominator = left-2*mid+right
                delta = np.divide(.5*(left-right), denominator, out=np.zeros_like(mid),
                                  where=np.abs(denominator)>1e-12)
                peak_offset = offsets[k]+np.clip(delta, -.5, .5)*.05
                alignment = mid / np.maximum(np.hypot(dx[ii, k], dy[ii, k]), 1e-12)
                valid &= alignment >= .8
                valid &= ((positions[:, :, 0].min(axis=1) >= 2) &
                          (positions[:, :, 0].max(axis=1) < image.shape[1]-2) &
                          (positions[:, :, 1].min(axis=1) >= 2) &
                          (positions[:, :, 1].max(axis=1) < image.shape[0]-2))
                observation = center + peak_offset[:, None]*normal
                rh = np.sum((h-observation)*normal, axis=1)[valid]
                rp = np.sum((p-observation)*normal, axis=1)[valid]
                groups.append({"reference_index": ref["index"], "model": ref["model"],
                               "component": component, "total": len(model), "valid": int(valid.sum()),
                               "halcon": stats(rh), "reproduction": stats(rp)})
                all_h.extend(rh); all_p.extend(rp); total += len(model)
        result["experiments"].append({"sigma": sigma, "total": total, "valid": len(all_h),
                                      "halcon": stats(np.asarray(all_h)),
                                      "reproduction": stats(np.asarray(all_p)), "groups": groups})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.predictions, args.reference), indent=2, allow_nan=False))
