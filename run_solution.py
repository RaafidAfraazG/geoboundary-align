#!/usr/bin/env python3
"""Boundary-raster alignment solution for the BhuMe take-home.

The starter baseline learns one village-wide translation from the public example
truths. This script keeps that baseline intact, then asks a local question for
each plot: does a small additional x/y shift put the polygon edge on stronger
field-boundary pixels?

The method is intentionally conservative. It writes a row for every plot, but
only marks a plot as ``corrected`` when the boundary raster gives a clear,
modest, and unambiguous improvement. Otherwise it writes ``flagged`` and keeps
the original official geometry.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from bhume import load, score, write_predictions
from bhume.baseline import global_median_shift

DEFAULT_VILLAGE = "data/34855_vadnerbhairav_chandavad_nashik"


def utm_for(geom: BaseGeometry) -> str:
    lon = geom.centroid.x
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1}"


def estimate_global_shift(village, utm: str) -> tuple[gpd.GeoDataFrame, float, float, int]:
    """Return a baseline GeoDataFrame in UTM plus the median shift in metres."""
    official_u = village.plots.to_crs(utm)
    if village.example_truths is None:
        baseline = official_u.copy()
        baseline["status"] = "flagged"
        baseline["confidence"] = 0.0
        baseline["method_note"] = "no example truths; started from official geometry"
        return baseline, 0.0, 0.0, 0

    truth_u = village.example_truths.to_crs(utm)
    dxs, dys = [], []
    for pn in village.example_truths.index:
        if pn in official_u.index:
            o = official_u.loc[pn, "geometry"].centroid
            t = truth_u.loc[pn, "geometry"].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
    if not dxs:
        raise ValueError("no overlapping plots between example truths and input.geojson")

    # Use the starter kit baseline for the actual initial geometry.
    baseline = global_median_shift(village).to_crs(utm)
    return baseline, statistics.median(dxs), statistics.median(dys), len(dxs)


def iter_lines(geom: BaseGeometry):
    if geom.is_empty:
        return
    boundary = geom.boundary
    if isinstance(boundary, LineString):
        yield boundary
    elif isinstance(boundary, MultiLineString):
        yield from boundary.geoms
    elif hasattr(boundary, "geoms"):
        for part in boundary.geoms:
            if isinstance(part, LineString):
                yield part
            elif isinstance(part, MultiLineString):
                yield from part.geoms


def edge_points(geom: BaseGeometry, spacing_m: float = 5.0, max_points: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Sample approximately even points along a polygon boundary in a metre CRS."""
    lines = [line for line in iter_lines(geom) if line.length > 0]
    total_len = sum(line.length for line in lines)
    if total_len <= 0:
        return np.array([]), np.array([])

    target_n = max(24, min(max_points, int(total_len / spacing_m)))
    xs, ys = [], []
    for line in lines:
        n = max(3, int(round(target_n * line.length / total_len)))
        distances = np.linspace(0, line.length, n, endpoint=False)
        for dist in distances:
            pt = line.interpolate(float(dist))
            xs.append(pt.x)
            ys.append(pt.y)

    if len(xs) > max_points:
        keep = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs = [xs[i] for i in keep]
        ys = [ys[i] for i in keep]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


@dataclass
class BoundaryPatch:
    distances_m: np.ndarray
    transform: object
    signal_fraction: float

    def score_points(
        self,
        xs_utm: np.ndarray,
        ys_utm: np.ndarray,
        to_raster: Transformer,
        dx_m: float = 0.0,
        dy_m: float = 0.0,
        sigma_m: float = 5.0,
        near_m: float = 4.0,
    ) -> float:
        if len(xs_utm) < 12 or self.signal_fraction <= 0:
            return 0.0

        xs_r, ys_r = to_raster.transform(xs_utm + dx_m, ys_utm + dy_m)
        inv = ~self.transform
        cols = inv.a * xs_r + inv.b * ys_r + inv.c
        rows = inv.d * xs_r + inv.e * ys_r + inv.f
        cols_i = np.rint(cols).astype(int)
        rows_i = np.rint(rows).astype(int)

        inside = (
            (rows_i >= 0)
            & (cols_i >= 0)
            & (rows_i < self.distances_m.shape[0])
            & (cols_i < self.distances_m.shape[1])
        )
        if inside.sum() < max(8, int(0.6 * len(xs_utm))):
            return 0.0

        dists = self.distances_m[rows_i[inside], cols_i[inside]]
        closeness = np.exp(-dists / sigma_m)
        coverage = np.mean(dists <= near_m)
        return float(0.75 * np.mean(closeness) + 0.25 * coverage)


class BoundaryAlignment:
    """Local distance-to-boundary scorer for a boundaries.tif raster."""

    def __init__(self, path: Path):
        self.src = rasterio.open(path)

    def close(self) -> None:
        self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _boundary_mask(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.zeros(arr.shape, dtype=bool)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.zeros(arr.shape, dtype=bool)

        nonzero = valid[valid > 0]
        if nonzero.size == 0:
            return np.zeros(arr.shape, dtype=bool)

        # Sparse binary-ish rasters should keep every positive hint. Dense
        # probability rasters use only the stronger quartile of non-zero pixels.
        if nonzero.size / valid.size < 0.20 or float(nonzero.max()) <= 1.0:
            return arr > 0
        threshold = float(np.percentile(nonzero, 75))
        return arr >= threshold

    def patch_for(
        self,
        geoms_utm: list[BaseGeometry],
        from_utm: Transformer,
        pad_m: float,
    ) -> BoundaryPatch | None:
        ras_geoms = [
            shp_transform(lambda xs, ys, z=None: from_utm.transform(xs, ys), geom)
            for geom in geoms_utm
            if geom is not None and not geom.is_empty
        ]
        if not ras_geoms:
            return None

        minx = min(g.bounds[0] for g in ras_geoms) - pad_m
        miny = min(g.bounds[1] for g in ras_geoms) - pad_m
        maxx = max(g.bounds[2] for g in ras_geoms) + pad_m
        maxy = max(g.bounds[3] for g in ras_geoms) + pad_m

        left = max(minx, self.src.bounds.left)
        bottom = max(miny, self.src.bounds.bottom)
        right = min(maxx, self.src.bounds.right)
        top = min(maxy, self.src.bounds.top)
        if right <= left or top <= bottom:
            return None

        window = from_bounds(left, bottom, right, top, transform=self.src.transform)
        arr = self.src.read(1, window=window, boundless=False)
        if arr.size == 0:
            return None

        mask = self._boundary_mask(arr)
        if not mask.any():
            return BoundaryPatch(np.full(arr.shape, 999.0, dtype=float), self.src.window_transform(window), 0.0)

        transform = self.src.window_transform(window)
        pixel_x = abs(transform.a)
        pixel_y = abs(transform.e)
        distances = distance_transform_edt(~mask, sampling=(pixel_y, pixel_x)).astype(float)
        return BoundaryPatch(distances, transform, float(mask.mean()))


@dataclass
class Candidate:
    dx: float
    dy: float
    edge_score: float
    objective: float


def shift_grid(max_shift_m: float, coarse_step_m: float, fine_step_m: float) -> list[tuple[float, float]]:
    coarse = np.arange(-max_shift_m, max_shift_m + 0.1, coarse_step_m)
    return [(float(dx), float(dy)) for dx in coarse for dy in coarse]


def refine_grid(best: Candidate, max_shift_m: float, fine_step_m: float) -> list[tuple[float, float]]:
    offsets = np.arange(-2 * fine_step_m, 2 * fine_step_m + 0.1, fine_step_m)
    pairs = []
    for ox in offsets:
        for oy in offsets:
            dx = float(best.dx + ox)
            dy = float(best.dy + oy)
            if math.hypot(dx, dy) <= max_shift_m + 0.01:
                pairs.append((dx, dy))
    return pairs


def choose_shift(
    patch: BoundaryPatch,
    baseline_xs: np.ndarray,
    baseline_ys: np.ndarray,
    to_raster: Transformer,
    max_shift_m: float,
    coarse_step_m: float,
    fine_step_m: float,
) -> tuple[Candidate, Candidate, float]:
    """Search local translations around the baseline polygon."""

    def eval_pair(dx: float, dy: float) -> Candidate:
        edge_score = patch.score_points(baseline_xs, baseline_ys, to_raster, dx, dy)
        shift_frac = min(1.0, math.hypot(dx, dy) / max_shift_m)
        objective = edge_score - 0.025 * shift_frac * shift_frac
        return Candidate(dx, dy, edge_score, objective)

    tried: dict[tuple[float, float], Candidate] = {}
    for dx, dy in shift_grid(max_shift_m, coarse_step_m, fine_step_m):
        tried[(round(dx, 3), round(dy, 3))] = eval_pair(dx, dy)

    coarse_best = max(tried.values(), key=lambda c: c.objective)
    for dx, dy in refine_grid(coarse_best, max_shift_m, fine_step_m):
        tried.setdefault((round(dx, 3), round(dy, 3)), eval_pair(dx, dy))

    ordered = sorted(tried.values(), key=lambda c: c.objective, reverse=True)
    best = ordered[0]
    meaningfully_different = [
        cand for cand in ordered[1:] if math.hypot(cand.dx - best.dx, cand.dy - best.dy) >= fine_step_m
    ]
    runner_up = meaningfully_different[0] if meaningfully_different else ordered[min(1, len(ordered) - 1)]
    margin = max(0.0, best.objective - runner_up.objective)
    return best, runner_up, margin


def boundary_signal_quality(signal_fraction: float) -> float:
    """Down-weight patches with too little signal or overly dense/noisy hints."""
    if signal_fraction <= 0:
        return 0.0
    enough_signal = min(1.0, signal_fraction / 0.015)
    not_too_dense = min(1.0, 0.12 / signal_fraction)
    return math.sqrt(enough_signal * not_too_dense)


def confidence_from(
    best: Candidate,
    gain_over_start: float,
    margin: float,
    max_shift_m: float,
    signal_quality: float,
    corrected: bool,
) -> float:
    shift_frac = min(1.0, math.hypot(best.dx, best.dy) / max_shift_m)
    score_strength = min(1.0, max(0.0, best.edge_score - 0.18) / 0.55)
    gain_strength = min(1.0, max(0.0, gain_over_start) / 0.22)
    margin_strength = min(1.0, margin / 0.035)
    shift_strength = max(0.0, 1.0 - shift_frac**1.4)

    if not corrected:
        conf = 0.08 + 0.20 * score_strength + 0.12 * gain_strength + 0.08 * margin_strength
        conf *= max(0.35, signal_quality)
        return round(max(0.0, min(0.49, conf)), 3)

    conf = (
        0.42
        + 0.14 * score_strength
        + 0.12 * gain_strength
        + 0.30 * margin_strength
        + 0.08 * signal_quality
        + 0.08 * shift_strength
    )
    if shift_frac > 0.55 and margin_strength < 0.35:
        conf = min(conf, 0.72)
    if signal_quality < 0.45:
        conf = min(conf, 0.68)
    return round(max(0.5, min(0.92, conf)), 3)


def build_predictions(
    village_dir: str | Path,
    max_shift_m: float = 18.0,
    coarse_step_m: float = 6.0,
    fine_step_m: float = 2.0,
) -> gpd.GeoDataFrame:
    village = load(village_dir)
    utm = utm_for(village.plots.geometry.iloc[0])
    official_u = village.plots.to_crs(utm)
    baseline_u, mdx, mdy, n_shift = estimate_global_shift(village, utm)

    rows = []
    corrected = 0
    flagged = 0
    no_boundary = village.boundaries_path is None

    if no_boundary:
        for pn, row in official_u.iterrows():
            rows.append(
                {
                    "plot_number": str(pn),
                    "status": "flagged",
                    "confidence": 0.0,
                    "method_note": "flagged: no boundaries.tif available for local alignment",
                    "geometry": row.geometry,
                }
            )
        return gpd.GeoDataFrame(rows, crs=utm).to_crs("EPSG:4326")

    with rasterio.open(village.boundaries_path) as src:
        boundary_crs = src.crs
    to_raster = Transformer.from_crs(utm, boundary_crs, always_xy=True)
    from_utm = to_raster

    with BoundaryAlignment(village.boundaries_path) as aligner:
        for i, pn in enumerate(official_u.index, start=1):
            official_geom = official_u.loc[pn, "geometry"]
            baseline_geom = baseline_u.loc[pn, "geometry"] if pn in baseline_u.index else official_geom
            baseline_xs, baseline_ys = edge_points(baseline_geom)
            official_xs, official_ys = edge_points(official_geom)

            patch = aligner.patch_for(
                [
                    official_geom,
                    baseline_geom.buffer(max_shift_m + 8.0),
                ],
                from_utm=from_utm,
                pad_m=max_shift_m + 12.0,
            )

            if patch is None or patch.signal_fraction <= 0 or len(baseline_xs) < 12:
                confidence = 0.0
                rows.append(
                    {
                        "plot_number": str(pn),
                        "status": "flagged",
                        "confidence": confidence,
                        "method_note": "flagged: insufficient boundary signal near plot",
                        "geometry": official_geom,
                    }
                )
                flagged += 1
                continue

            official_score = patch.score_points(official_xs, official_ys, to_raster)
            baseline_score = patch.score_points(baseline_xs, baseline_ys, to_raster)
            best, runner_up, margin = choose_shift(
                patch,
                baseline_xs,
                baseline_ys,
                to_raster,
                max_shift_m=max_shift_m,
                coarse_step_m=coarse_step_m,
                fine_step_m=fine_step_m,
            )

            gain_over_baseline = best.edge_score - baseline_score
            gain_over_official = best.edge_score - official_score
            gain_over_start = best.edge_score - max(baseline_score, official_score)
            local_shift_m = math.hypot(best.dx, best.dy)
            signal_quality = boundary_signal_quality(patch.signal_fraction)

            accept = (
                best.edge_score >= 0.18
                and signal_quality >= 0.25
                and gain_over_baseline >= 0.025
                and gain_over_official >= 0.015
                and margin >= 0.004
                and local_shift_m <= max_shift_m
            )
            if official_score >= 0.50 and gain_over_official < 0.08:
                accept = False
            if baseline_score >= 0.60 and gain_over_baseline < 0.05:
                accept = False
            if local_shift_m > 0.55 * max_shift_m and margin < 0.008:
                accept = False
            if local_shift_m > 0.75 * max_shift_m and margin < 0.014:
                accept = False

            confidence = confidence_from(
                best,
                gain_over_start,
                margin,
                max_shift_m,
                signal_quality,
                corrected=accept,
            )
            if accept:
                status = "corrected"
                geom = translate(baseline_geom, best.dx, best.dy)
                corrected += 1
                note = (
                    f"boundary alignment: global dx={mdx:.1f} dy={mdy:.1f}m from {n_shift} truths; "
                    f"local dx={best.dx:.1f} dy={best.dy:.1f}m; "
                    f"score {best.edge_score:.3f} vs base {baseline_score:.3f}, official {official_score:.3f}; "
                    f"margin {margin:.3f}; signal {signal_quality:.2f}"
                )
            else:
                status = "flagged"
                geom = official_geom
                flagged += 1
                note = (
                    f"flagged: weak or ambiguous boundary alignment; "
                    f"best {best.edge_score:.3f}, base {baseline_score:.3f}, official {official_score:.3f}, "
                    f"margin {margin:.3f}, local shift {local_shift_m:.1f}m, signal {signal_quality:.2f}"
                )

            rows.append(
                {
                    "plot_number": str(pn),
                    "status": status,
                    "confidence": confidence,
                    "method_note": note,
                    "geometry": geom,
                }
            )

            if i % 250 == 0:
                print(f"  processed {i}/{len(official_u)} plots ({corrected} corrected, {flagged} flagged)")

    return gpd.GeoDataFrame(rows, crs=utm).to_crs("EPSG:4326")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative boundary-raster alignment.")
    parser.add_argument("village_dir", nargs="?", default=DEFAULT_VILLAGE)
    parser.add_argument("--output", default=None, help="Output predictions.geojson path")
    parser.add_argument("--max-shift-m", type=float, default=18.0)
    parser.add_argument("--coarse-step-m", type=float, default=6.0)
    parser.add_argument("--fine-step-m", type=float, default=2.0)
    args = parser.parse_args()

    village = load(args.village_dir)
    print(f"Loaded {village.slug}")
    print(
        f"  {len(village.plots)} plots; "
        f"boundaries={'yes' if village.boundaries_path else 'none'}; "
        f"example_truths={0 if village.example_truths is None else len(village.example_truths)}"
    )

    preds = build_predictions(
        args.village_dir,
        max_shift_m=args.max_shift_m,
        coarse_step_m=args.coarse_step_m,
        fine_step_m=args.fine_step_m,
    )
    out = write_predictions(args.output or (Path(args.village_dir) / "predictions.geojson"), preds)
    print(f"  wrote {len(preds)} predictions -> {out}")
    print(f"  corrected={(preds.status == 'corrected').sum()} flagged={(preds.status == 'flagged').sum()}")

    if village.example_truths is not None:
        print()
        print(score(preds, village))


if __name__ == "__main__":
    main()
