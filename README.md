# CV Algorithms

Computer vision algorithm experiments and reproducible prototypes.

## Projects

### Shape Matching

`shape_match` is a C++17 + OpenCV library for HALCON-like shape-based
matching. Version 0.3.0 covers gradient-orientation models, image
pyramids, joint rotation/scale candidates, coarse-to-fine search,
multi-object NMS, and optional subpixel position refinement.

- [Project README](shape_match/README.md)
- [Implementation plan](shape_match/plan.md)
- [0.1 final handoff](shape_match/docs/final-handoff.md)
- [0.2 scale-search handoff](shape_match/docs/scale-search-handoff.md)

## Repository Status

The 0.2 scale-search implementation and release baseline are complete. It
includes CTest smoke/unit/integration/package-consumer coverage, percentile
benchmarks, install/export support, deterministic multithreading, and a
HALCON-like compatibility matrix. Polarity, persistence, and GPU features
remain separate follow-up projects.
