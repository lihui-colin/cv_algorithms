"""Template matching with Tiny RoMa.

The template and target may have different sizes. Tiny RoMa produces dense,
normalized correspondences; this script samples them, estimates a homography,
and projects the four template corners into the target image.

Example:
    python roma_template_match.py \
        --template template.jpg --target scene.jpg \
        --output scene_match.jpg --json-output scene_match.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _add_local_roma_to_path() -> None:
    """Allow running from this checkout before installing RoMa editable."""
    try:
        import romatch  # noqa: F401
        return
    except ImportError:
        local_repo = Path(__file__).resolve().parent / "RoMa"
        if (local_repo / "romatch").is_dir():
            sys.path.insert(0, str(local_repo))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but MPS is unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tiny_roma_model(device: torch.device):
    """Load Tiny RoMa, reusing the cached checkpoint without GitHub API access.

    The upstream factory also calls ``torch.hub.load`` for XFeat and first
    queries the GitHub API. The Tiny RoMa checkpoint already contains the XFeat
    backbone, so Kornia's compatible XFeatModel is sufficient and avoids that
    extra network request once the Tiny checkpoint has been downloaded.
    """
    from romatch import tiny_roma_v1_outdoor

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "tiny_roma_v1_outdoor.pth"
    if not checkpoint.is_file():
        return tiny_roma_v1_outdoor(device=device)

    from kornia.feature import XFeatModel
    from romatch.models.model_zoo.roma_models import tiny_roma_v1_model

    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    return tiny_roma_v1_model(weights=weights, xfeat=XFeatModel()).to(device)


def estimate_template_homography(
    model,
    template_path: Path,
    target_path: Path,
    num_matches: int,
    certainty_threshold: float,
    ransac_threshold: float,
):
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Cannot read template image: {template_path}")
    if target is None:
        raise FileNotFoundError(f"Cannot read target image: {target_path}")
    ht, wt = template.shape[:2]
    hi, wi = target.shape[:2]

    # RoMa accepts paths directly and returns [x_t, y_t, x_i, y_i] in [-1, 1].
    warp, certainty = model.match(str(template_path), str(target_path))
    certainty = certainty.reshape(-1)
    num = min(num_matches, certainty.numel())
    if num < 4:
        raise RuntimeError("Not enough dense correspondences were produced")

    # Top-certainty points are deterministic and avoid multinomial failures on
    # images where all certainties are very small. Keep a broad spatial spread
    # by asking RoMa's balanced sampler first, then fall back to top-k.
    try:
        matches, sampled_certainty = model.sample(warp, certainty.reshape(warp.shape[:2]), num)
    except (RuntimeError, ValueError):
        flat_idx = torch.topk(certainty, k=num, largest=True).indices
        matches = warp.reshape(-1, 4)[flat_idx]
        sampled_certainty = certainty[flat_idx]

    keep = sampled_certainty >= certainty_threshold
    if int(keep.sum()) < 4:
        # A low-certainty pair can still be matchable; use the best four points
        # rather than failing with an opaque OpenCV error.
        best = torch.topk(sampled_certainty, k=min(4, sampled_certainty.numel())).indices
        keep = torch.zeros_like(sampled_certainty, dtype=torch.bool)
        keep[best] = True
    matches = matches[keep]
    sampled_certainty = sampled_certainty[keep]

    kpts_template, kpts_target = model.to_pixel_coordinates(
        matches, ht, wt, hi, wi
    )
    pts_t = kpts_template.detach().cpu().numpy().astype(np.float32)
    pts_i = kpts_target.detach().cpu().numpy().astype(np.float32)
    if len(pts_t) < 4:
        raise RuntimeError("At least four valid correspondences are required")

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, inlier_mask = cv2.findHomography(
        pts_t, pts_i, method, ransacReprojThreshold=ransac_threshold,
        maxIters=10000, confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Could not estimate a homography from RoMa matches")

    inliers = inlier_mask.ravel().astype(bool)
    corners = np.float32([[[0, 0], [wt - 1, 0], [wt - 1, ht - 1], [0, ht - 1]]])
    projected = cv2.perspectiveTransform(corners, homography)[0]
    projected_area = abs(float(cv2.contourArea(projected.reshape(-1, 1, 2))))
    reproj = cv2.perspectiveTransform(pts_t.reshape(-1, 1, 2), homography)[:, 0]
    reproj_error = np.linalg.norm(reproj - pts_i, axis=1)

    return {
        "homography": homography,
        "projected_corners": projected,
        "inlier_mask": inliers,
        "template_size": [wt, ht],
        "target_size": [wi, hi],
        "num_matches": int(len(pts_t)),
        "num_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "mean_inlier_certainty": float(sampled_certainty[inliers].mean().item())
        if inliers.any() else 0.0,
        "median_inlier_reprojection_error_px": float(np.median(reproj_error[inliers]))
        if inliers.any() else None,
        "projected_area_px": projected_area,
    }


def draw_result(target_path: Path, result: dict, output_path: Path) -> None:
    image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    polygon = np.round(result["projected_corners"]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    for i, (x, y) in enumerate(polygon[:, 0]):
        cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(image, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
    text = f"inliers: {result['num_inliers']}/{result['num_matches']} ({result['inlier_ratio']:.2f})"
    cv2.putText(image, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Cannot write output image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Template matching using Tiny RoMa")
    parser.add_argument("--template", required=True, type=Path, help="template image")
    parser.add_argument("--target", required=True, type=Path, help="larger/search image")
    parser.add_argument("--output", type=Path, default=Path("roma_template_match.jpg"))
    parser.add_argument("--json-output", type=Path, default=Path("roma_template_match.json"))
    parser.add_argument("--num-matches", type=int, default=2000)
    parser.add_argument("--certainty-threshold", type=float, default=0.2)
    parser.add_argument("--ransac-threshold", type=float, default=4.0,
                        help="RANSAC reprojection threshold in target pixels")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    _add_local_roma_to_path()
    device = choose_device(args.device)
    print(f"Loading Tiny RoMa on {device}; first run may download model weights...")
    model = load_tiny_roma_model(device)
    result = estimate_template_homography(
        model, args.template, args.target, args.num_matches,
        args.certainty_threshold, args.ransac_threshold,
    )
    draw_result(args.target, result, args.output)
    serializable = {k: v for k, v in result.items() if k not in {"homography", "projected_corners", "inlier_mask"}}
    serializable["homography"] = result["homography"].tolist()
    serializable["projected_corners"] = result["projected_corners"].tolist()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
