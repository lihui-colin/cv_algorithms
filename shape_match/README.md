# OpenShapeMatch

C++17 + OpenCV implementation of a HALCON-like shape-based matching library.
The public API is exposed from `include/openshape/openshape.hpp`; the current
package version is 0.3.0.

## Build

OpenCV 4.x development headers and libraries are required. From the
repository root:

```sh
cmake -S shape_match -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix /desired/prefix
```

Installed CMake consumers use:

```cmake
find_package(OpenShapeMatch 0.3 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openshape::OpenShapeMatch)
```

The `shape_match_example` executable accepts a scene image followed by a
template image. Scale search is configured with `SearchParams::scale_min`,
`scale_max`, and `scale_step`; their defaults keep fixed-scale `1.0` behavior.
Set `SearchParams::enable_subpixel` to enable optional subpixel position
refinement, or `enable_pose_refinement` for joint continuous position, angle,
and scale optimization. The precise path uses bilinear gradients and a soft
distance-to-edge response and reports convergence, residual, confidence, and
visible-point fraction in each `MatchResult`.

Polarity is selected with `SearchParams::polarity` (`Same`, `Inverted`,
`GlobalEither`, or `LocalEither`). Partial occlusion is controlled by
`min_visible_fraction`; its default `0.25` preserves the legacy acceptance
rule. All new behavior is opt-in except the now-configurable legacy threshold.

Models use a checksummed, explicitly little-endian native format:

```cpp
model.save("part.osm");
auto loaded = openshape::ShapeModel::load("part.osm");
```

The format stores model metadata, edge parameters, model points, weights, and
pyramid levels. It intentionally does not read or write HALCON model files.

To build the local benchmark target:

```sh
cmake -S shape_match -B build-benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$HOME/.local/opencv-4.10.0/lib/cmake/opencv4" \
  -DBUILD_BENCHMARK=ON -DBUILD_TESTING=OFF
cmake --build build-benchmark -j8
LD_LIBRARY_PATH="$HOME/.local/opencv-4.10.0/lib:${LD_LIBRARY_PATH:-}" \
  ./build-benchmark/shape_match_benchmark
```

The matcher supports deterministic candidate merging with
`SearchParams::num_threads`; `0` selects an automatic bounded worker count,
while a positive value requests a fixed worker count.

For large angle/scale ranges, the HALCON-style optimized CPU pipeline can be
enabled incrementally:

```cpp
search.enable_fast_pipeline();
```

The default is `strict_detection=true`: coarse reductions, Greediness, and
peak clustering are disabled, and every returned pose is rescored against the
complete model. This correctness-first mode should be used when missed or
false detections are unacceptable. `enable_fast_pipeline()` explicitly opts
into the throughput-oriented path; applications should validate its recall
and precision on their own target data before deployment.

The default values keep the historical full-point discrete search. Continuous
refinement always applies joint position/angle/scale peak selection before its
bounded Top-K optimization; set `refinement_candidate_limit=0` only when an
unlimited diagnostic search is explicitly required.
Implementation details and local measurements are in
[`docs/fast-pipeline.md`](docs/fast-pipeline.md).

For repeated searches over the same scene, compute an `EdgeMap` once with
`EdgeEngine::compute` and call the `find_shape_models(const EdgeMap&, ...)`
overload. The Phase 7A performance handoff and measured baseline are in
[`docs/phase-7-handoff.md`](docs/phase-7-handoff.md).

For repeated searches over the same scene and multiple pyramid levels, build
an `EdgePyramid` once and call `find_shape_models(const EdgePyramid&, ...)`.
Phase 7B also exposes portable CPU feature reporting and the benchmark matrix
in [`docs/phase-7b-handoff.md`](docs/phase-7b-handoff.md).

An AVX2 score kernel can be compiled with `OPENSHAPE_ENABLE_AVX2=ON` and
selected experimentally with `OPENSHAPE_USE_AVX2_BY_DEFAULT=ON`. It remains
opt-in because the current sparse-gather implementation is slower than the
portable SoA kernel in end-to-end benchmarks. See
[`docs/phase-7c-handoff.md`](docs/phase-7c-handoff.md).

The coarse global search now uses a conservative Canny coverage prefilter and
parallelizes angle-row jobs. Set `SearchParams::enable_coarse_prefilter` to
`false` for the exact unfiltered baseline, or pass `SearchStats*` to a matching
overload to inspect pose counts and phase timings. Details and benchmark data
are in [`docs/phase-7d-handoff.md`](docs/phase-7d-handoff.md).

The fixed-scale performance-stage results are in
[`docs/phase-7e-handoff.md`](docs/phase-7e-handoff.md). Release scope and API
compatibility are documented in [`docs/release-0.2.0.md`](docs/release-0.2.0.md),
[`docs/compatibility.md`](docs/compatibility.md), and the
[`scale-search handoff`](docs/scale-search-handoff.md).

## Scope

The first implementation milestone targets a reproducible coarse-to-fine
matching loop with:

- edge and gradient-orientation models;
- image and model pyramids;
- joint rotation and isotropic scale candidate search;
- multi-object detection and non-maximum suppression (NMS);
- configurable score, match-count, and overlap thresholds.

The implementation aims to provide compatible algorithm semantics and
parameters where practical. It does not claim to reproduce HALCON's
undisclosed internal implementation.

## Planned Structure

| Path | Purpose |
| --- | --- |
| `shape_match/` | Model, pyramid, scoring, search, and result-processing modules |
| `data/` | Synthetic or sample input data |
| `gt/` | Ground-truth annotations |
| `scripts/` | Experiment and evaluation scripts |
| `tests/` | Regression tests and synthetic test cases |
| `examples/` | Model creation, matching, and visualization examples |

## Development Milestones

1. Define the C++ library, dependencies, and core data structures.
2. Create gradient-based shape models and model pyramids.
3. Implement pyramid search with translation and rotation support.
4. Add score filtering, overlap suppression, and result visualization.
5. Add synthetic data and regression tests.
6. Extend the baseline with polarity, occlusion handling, and model
  persistence and continuous four-dimensional pose refinement while keeping
  the historical discrete mode as the default.

The detailed plan, verification criteria, and design decisions are in
[plan.md](plan.md).

## Test Data

| File | Description | Source |
| --- | --- | --- |
| `data/invariant_image_1.png` | Metal-part scene with a rotated target and a second distractor | [`Invariant-TemplateMatching/images/image_1.png`](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/images/image_1.png) |
| `data/invariant_template_1.jpg` | Template corresponding to the target in `invariant_image_1.png` | [`Invariant-TemplateMatching/images/template_1.jpg`](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/images/template_1.jpg) |
| `data/invariant_image_2.jpg` | Fruit scene with multiple objects and varied colors | [`Invariant-TemplateMatching/images/image_2.jpg`](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/images/image_2.jpg) |
| `data/invariant_template_2.jpg` | Template corresponding to one fruit in `invariant_image_2.jpg` | [`Invariant-TemplateMatching/images/template_2.jpg`](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/images/template_2.jpg) |

All four files are `640 x 400` images downloaded from the
[`Invariant-TemplateMatching`](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching)
repository, which is distributed under the
[MIT License](https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/LICENSE).

## References

- [HALCON shape-based matching documentation](https://www.mvtec.com/doc/halcon/13/en/toc_matching_shapebased.html)
- [Open-source shape-based matching reference](https://github.com/meiqua/shape_based_matching)
- [Shape matching paper](https://ieeexplore.ieee.org/document/4582953/)
