"""Check that diagnostic instrumentation preserves production results."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sample, probe, data = map(Path, sys.argv[1:])
with tempfile.TemporaryDirectory(prefix="shape-trajectory-test-") as tmp:
    root = Path(tmp)
    subprocess.run([str(sample), "--output-dir", str(root / "sample")], cwd=data.parent, check=True,
                   capture_output=True, text=True)
    subprocess.run([str(probe), str(data), str(root / "trace")], check=True, capture_output=True, text=True)
    def read(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    a, b = read(root / "sample/matching_results.csv"), read(root / "trace.csv")
    assert len(a) == len(b) == 7
    for left, right in zip(a, b):
        assert left["model"] == right["model"]
        for key in ("row", "column", "angle_deg", "scale_row", "scale_column", "score"):
            assert abs(float(left[key])-float(right[key])) < 1e-9, key
    traces = read(root / "trace.trace.csv")
    ids = {r["trace_id"] for r in traces}
    for index in ids:
        events = [r for r in traces if r["trace_id"] == index]
        assert events[0]["event"] == "begin" and events[-1]["event"] == "end"
    subprocess.run([str(probe), str(data), str(root / "zero"), "0"], check=True, capture_output=True, text=True)
    traces = read(root / "zero.trace.csv")
    for index in {r["trace_id"] for r in traces}:
        events = [r for r in traces if r["trace_id"] == index]
        assert len(events) == 2
        for key in ("row", "column", "angle_deg", "scale"):
            assert events[0][key] == events[1][key]
    # Import our own output, not external GT: exercises public/internal pose
    # conversion with nonzero rotation and non-unit scale on every candidate.
    for residual in ([], ["point_to_point"]):
        prefix = root / ("seed-point" if residual else "seed-normal")
        run = subprocess.run([str(probe), str(data), str(prefix), "0", "0",
                              str(root / "trace.csv"), *residual], check=True,
                             capture_output=True, text=True)
        assert "not a standalone detector evaluation" in run.stderr
        seeded = read(prefix.with_suffix(".trace.csv"))
        for event in seeded:
            expected = [r for r in b if r["model"] == event["model"] and
                        (float(r["row"])-float(event["row"]))**2 +
                        (float(r["column"])-float(event["column"]))**2 < 9]
            assert len(expected) == 1
            for key, source in (("row", "row"), ("column", "column"),
                                ("angle_deg", "angle_deg"), ("scale", "scale_row")):
                assert abs(float(event[key])-float(expected[0][source])) < 1e-9, key
    for policy in ("frozen", "support"):
        prefix = root / policy
        subprocess.run([str(probe), str(data), str(prefix), "30", "1.5"],
                       env={**os.environ, "SHAPE_MATCH_TRACE_UPDATE": policy},
                       capture_output=True, text=True, check=True)
        updates = read(prefix.with_suffix(".updates.csv"))
        assert updates
        for row in updates:
            if row["accepted"] == "1":
                assert float(row["frozen_trial_loss"]) <= float(row["loss_before"]) + 1e-8
                if policy == "support":
                    assert float(row["support_after"]) <= float(row["support_before"]) + 1e-8
        if policy == "frozen":
            for trace_id in {r["trace_id"] for r in updates}:
                assert len({r["count"] for r in updates if r["trace_id"] == trace_id}) == 1
    run = subprocess.run([str(probe), str(data), str(root / "invalid")],
                         env={**os.environ, "SHAPE_MATCH_TRACE_UPDATE": "invalid"},
                         capture_output=True, text=True)
    assert run.returncode != 0 and "Unknown diagnostic update policy" in run.stderr
    for arguments, message in ((["0", "nan"], "Radius"),
                               (["0", "0", str(root / "missing.csv")], "Cannot read"),
                               (["0", "0", "native", "invalid"], "Unknown residual")):
        run = subprocess.run([str(probe), str(data), str(root / "invalid"), *arguments],
                             capture_output=True, text=True)
        assert run.returncode != 0 and message in run.stderr
print("PASS: production equivalence, trace boundaries, seeds, update policies and argument validation")
