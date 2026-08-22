# CPU fast matching pipeline

The optimized path mirrors the publicly documented structure of industrial
coarse-to-fine shape matchers while retaining the historical search as the
default:

1. strongest-point reduction on successively coarser model levels;
2. Canny coverage prefilter;
3. safe orientation-score upper-bound termination (`greediness`);
4. position/angle/scale joint peak clustering between pyramid levels;
5. bounded Top-K continuous refinement with several starts per peak;
6. final full-point validation and overlap suppression.

Enable the conservative preset with:

```cpp
openshape::SearchParams search;
search.enable_fast_pipeline();
```

For inspection, `SearchParams::strict_detection` defaults to `true`. In this
mode all returned poses are rescored using the complete finest-level model and
the potentially recall-losing optimizations are bypassed. The preset sets it
to `false` explicitly because its purpose is throughput benchmarking; it must
not be treated as a correctness guarantee without a dataset-specific recall
and false-positive evaluation.

The preset selects `greediness=0.9`, `coarse_point_fraction=0.75`, joint
candidate clustering, and a 32-candidate refinement limit. Applications with
many closely spaced targets should increase `coarse_candidate_limit` and
`refinement_candidate_limit`. Setting `coarse_point_fraction=1.0` disables
point reduction when maximum recall is more important than latency.

The model builder honors `model_point_sampling`: `uniform` (the default) uses
spatially distributed points with equal confidence, while `magnitude` retains
gradient-strength weighting. This prevents a few bright texture highlights
from dominating natural-object matches.

## Local Release measurements

Intel Xeon Platinum 8368Q, GCC 13.3, OpenCV 4.10, four matcher threads:

| Scenario | Baseline p50 | Optimized p50 | Result |
| --- | ---: | ---: | --- |
| Real sample 1, 61 angles x 9 scales | 564 ms | 283 ms | Same top pose and score |
| Synthetic continuous refinement | 1084 ms | 47 ms | 1029 candidates reduced to 32 |
| 640x400 clutter, 25 transforms, full call (smoke) | 51.4 ms | 33.2 ms | Same detected scale |

The synthetic refinement retains approximately the same measured pose error;
this work improves candidate throughput, not the remaining subpixel position
bias. Formal percentile measurements should use at least 30 iterations on the
deployment CPU.
