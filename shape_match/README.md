# Shape Matching

Python + OpenCV implementation of a HALCON-like shape-based matching
prototype.

## Scope

The first implementation milestone targets a reproducible coarse-to-fine
matching loop with:

- edge and gradient-orientation models;
- image and model pyramids;
- rotation candidate search;
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

1. Define the Python package, dependencies, and core data structures.
2. Create gradient-based shape models and model pyramids.
3. Implement pyramid search with translation and rotation support.
4. Add score filtering, overlap suppression, and result visualization.
5. Add synthetic data and regression tests.
6. Extend the baseline with scale, polarity, subpixel refinement, occlusion
   handling, and model persistence.

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
