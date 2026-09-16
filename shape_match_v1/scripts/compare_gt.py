#!/usr/bin/env python3
"""Compare detector CSVs without passing reference data to the detector (stdlib only)."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        row["index"] = int(row["index"])
        if row["index"] in seen:
            raise ValueError("Duplicate result index")
        seen.add(row["index"])
        for key in ("row", "column", "angle_deg", "scale_row", "scale_column", "score"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite {key}")
        if min(row["scale_row"], row["scale_column"]) <= 0:
            raise ValueError("Scale must be positive")
    return rows


def compare(predictions, references, gate=3.0, bundled_labels=False):
    if not math.isfinite(gate) or gate <= 0:
        raise ValueError("Association gate must be finite and positive")
    if not references:
        raise ValueError("Empty reference set is not an accuracy test")
    if bundled_labels and {r["index"] for r in references} != set(range(7)):
        raise ValueError("Bundled label correction requires reference indices 0..6")
    used, comparisons = set(), []
    for ref in references:
        name = ("ring" if ref["index"] in (1, 2, 5) else "nut") if bundled_labels else ref["model"]
        candidates = []
        for i, pred in enumerate(predictions):
            if pred["model"] != name:
                continue
            distance = math.hypot(pred["row"] - ref["row"], pred["column"] - ref["column"])
            if distance < gate:
                candidates.append((i, distance))
        # The bundled instances are far apart. Fail on ambiguity rather than
        # silently change association when a method emits duplicate detections.
        if len(candidates) > 1 or (candidates and candidates[0][0] in used):
            raise ValueError(f"Ambiguous association for reference {ref['index']}")
        item = {"reference_index": ref["index"], "model": name, "matched": bool(candidates)}
        if candidates:
            i, distance = candidates[0]
            used.add(i)
            pred = predictions[i]
            angular = pred["angle_deg"] - ref["angle_deg"]
            period = (45 if name == "ring" else 60) if bundled_labels else 360
            item.update(prediction_index=pred["index"], position_error_px=distance,
                        row_error_px=pred["row"] - ref["row"],
                        column_error_px=pred["column"] - ref["column"],
                        angle_error_raw_deg=angular,
                        angle_error_mod_symmetry_deg=math.remainder(angular, period),
                        scale_row_relative_error=pred["scale_row"] / ref["scale_row"] - 1,
                        scale_column_relative_error=pred["scale_column"] / ref["scale_column"] - 1)
        comparisons.append(item)
    errors = sorted(r["position_error_px"] for r in comparisons if r["matched"])
    return {"expected": len(references), "detected": len(predictions), "associated": len(errors),
            "extra": len(predictions) - len(used), "bundled_label_correction": bundled_labels,
            "position_rmse_px": math.sqrt(sum(x*x for x in errors) / len(errors)) if errors else None,
            "position_max_px": max(errors) if errors else None,
            "position_p95_px": errors[math.ceil(.95 * len(errors)) - 1] if errors else None,
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "data/reference/matching_results.csv")
    parser.add_argument("--bundled-labels", action="store_true", help="Apply the documented historical labels for the bundled GT only")
    parser.add_argument("--gate", type=float, default=3)
    parser.add_argument("--max-rmse", type=float)
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for threshold in (args.max_rmse, args.max_error):
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            parser.error("Precision thresholds must be finite and nonnegative")
    try:
        result = compare(read_rows(args.prediction), read_rows(args.reference), args.gate, args.bundled_labels)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    associated = result["associated"] == result["expected"] and result["extra"] == 0
    requested = args.max_rmse is not None or args.max_error is not None
    passes = associated
    for key, threshold in (("position_rmse_px", args.max_rmse), ("position_max_px", args.max_error)):
        if threshold is not None:
            passes = passes and result[key] is not None and result[key] <= threshold
    result["association_passed"] = associated
    result["precision_passed"] = passes if requested else None
    result["limits"] = {"rmse_px": args.max_rmse, "max_px": args.max_error}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
