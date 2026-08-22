# OpenShapeMatch 0.3.0

This release adds an opt-in CPU high-accuracy path without changing the
historical discrete-search defaults.

## Added

- Bilinear gradient and soft distance-to-edge scoring through
  `score_pose_precise`.
- Joint column, row, angle, and scale refinement with bounded quadratic local
  fits and discrete-pose fallback.
- Same, inverted, global-either, and local-either polarity modes.
- Configurable visible-model-point threshold for partial occlusion.
- Refinement convergence, residual, confidence, and valid-point fraction in
  `MatchResult`, plus refinement timing/evaluation statistics.
- Checksummed, versioned, explicit little-endian `.osm` model persistence.
- Benchmark output for compiler/OpenCV/CPU metadata, p50/p95/p99, candidate
  counts, phase timing, and peak resident memory.
- Safe Greediness score bounds, coarse-level strongest-point reduction,
  position/angle/scale peak clustering, and bounded Top-K continuous
  refinement.

## Compatibility

`enable_subpixel` and `enable_pose_refinement` are false by default,
`PolarityMode::Same` is the default, and `min_visible_fraction=0.25` matches
the former fixed acceptance rule. Existing score overloads and result ordering
remain available.

GPU, Python bindings, HALCON file interoperability, and an independently
licensed HALCON timing comparison are outside this release.
