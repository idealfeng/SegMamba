import argparse
import csv
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal


NDSI_FILL = 255
STATIC_FILL = -9999


# ----------------------------
# Date helpers (fixed 181, drop Feb29)
# ----------------------------
def winter_dates(start_year: int) -> List[str]:
    d0 = date(start_year, 11, 1)
    d1 = date(start_year + 1, 4, 30)
    out = []
    cur = d0
    while cur <= d1:
        if not (cur.month == 2 and cur.day == 29):
            out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    # Hard contract: 181 always
    if len(out) != 181:
        raise ValueError(
            f"Expected 181 dates, got {len(out)} for start_year={start_year}"
        )
    return out


def has_feb29_in_season(start_year: int) -> bool:
    # Season covers Nov(start_year) to Apr(start_year+1)
    y = start_year + 1
    # Feb29 exists only on leap years
    try:
        _ = date(y, 2, 29)
        return True
    except ValueError:
        return False


# ----------------------------
# File grouping
# ----------------------------
GROUP_RE = re.compile(
    r"^(?P<gid>.+?_\d{4}_\d{4}_.+?)(?:-\d{10}-\d{10})?\.tif$", re.IGNORECASE
)


def group_id_from_path(p: str) -> Optional[str]:
    base = os.path.basename(p)
    m = GROUP_RE.match(base)
    return m.group("gid") if m else None


def years_from_group_id(gid: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_(\d{4})_(\d{4})_", gid)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ----------------------------
# Layout detection
# ----------------------------
@dataclass
class Layout:
    name: str
    per_day: int  # 2 or 3
    day_count: int
    static_count: int

    # Return (ndsi_band_index, mask_band_index, qmask_band_index_or_None) for day t
    def band_of_day(self, t: int) -> Tuple[int, int, Optional[int]]:
        raise NotImplementedError


class Layout2InterleavedNDSI_MASK(Layout):
    def band_of_day(self, t: int):
        return (2 * t + 1, 2 * t + 2, None)


class Layout2InterleavedMASK_NDSI(Layout):
    def band_of_day(self, t: int):
        # return (ndsi, mask)
        return (2 * t + 2, 2 * t + 1, None)


class Layout2GroupedNDSI_then_MASK(Layout):
    def band_of_day(self, t: int):
        return (t + 1, self.day_count + t + 1, None)


class Layout2GroupedMASK_then_NDSI(Layout):
    def band_of_day(self, t: int):
        return (self.day_count + t + 1, t + 1, None)


class Layout3InterleavedNDSI_QMASK_MASK(Layout):
    def band_of_day(self, t: int):
        base = 3 * t
        return (base + 1, base + 3, base + 2)  # ndsi, mask, qmask


class Layout3InterleavedNDSI_MASK_QMASK(Layout):
    def band_of_day(self, t: int):
        base = 3 * t
        return (base + 1, base + 2, base + 3)  # ndsi, mask, qmask


class Layout3InterleavedMASK_NDSI_QMASK(Layout):
    def band_of_day(self, t: int):
        base = 3 * t
        return (base + 2, base + 1, base + 3)


class Layout3InterleavedMASK_QMASK_NDSI(Layout):
    def band_of_day(self, t: int):
        base = 3 * t
        return (base + 3, base + 1, base + 2)


class Layout3GroupedNDSI_QMASK_MASK(Layout):
    def band_of_day(self, t: int):
        # [NDSI x D] [QMASK x D] [MASK x D]
        ndsi = t + 1
        qmask = self.day_count + t + 1
        mask = 2 * self.day_count + t + 1
        return (ndsi, mask, qmask)


class Layout3GroupedNDSI_MASK_QMASK(Layout):
    def band_of_day(self, t: int):
        # [NDSI x D] [MASK x D] [QMASK x D]
        ndsi = t + 1
        mask = self.day_count + t + 1
        qmask = 2 * self.day_count + t + 1
        return (ndsi, mask, qmask)


def is_binary(arr: np.ndarray) -> bool:
    u = np.unique(arr)
    return u.size <= 3 and set(u.tolist()).issubset({0, 1})


def looks_like_ndsi(arr: np.ndarray) -> bool:
    # NDSI should be integer, in [0,255], and usually contains 255
    if arr.dtype.kind not in "iu":
        return False
    mn, mx = int(arr.min()), int(arr.max())
    if mn < 0 or mx > 255:
        return False
    # Not purely binary
    if is_binary(arr):
        return False
    # Prefer if it contains 255, but allow some tiles with 0 valid?
    return (mx == 255) or (np.mean(arr == 255) > 1e-4)


def looks_like_qmask(arr: np.ndarray) -> bool:
    # QMASK should be small int set (0..3 typically)
    if arr.dtype.kind not in "iu":
        return False
    mn, mx = int(arr.min()), int(arr.max())
    if mn < 0 or mx > 10:
        return False
    u = np.unique(arr)
    return u.size <= 10


@dataclass(frozen=True)
class Window:
    col_off: int
    row_off: int
    width: int
    height: int


class GeoDataset:
    def __init__(self, ds: gdal.Dataset):
        self._ds = ds

    @property
    def width(self) -> int:
        return int(self._ds.RasterXSize)

    @property
    def height(self) -> int:
        return int(self._ds.RasterYSize)

    @property
    def count(self) -> int:
        return int(self._ds.RasterCount)

    def read(self, indexes, window: Optional[Window] = None) -> np.ndarray:
        if isinstance(indexes, (list, tuple)):
            return np.stack([self.read(i, window=window) for i in indexes], axis=0)

        idx = int(indexes)
        band = self._ds.GetRasterBand(idx)
        if band is None:
            raise ValueError(f"Invalid band index: {idx}")

        if window is None:
            arr = band.ReadAsArray()
        else:
            arr = band.ReadAsArray(
                xoff=int(window.col_off),
                yoff=int(window.row_off),
                win_xsize=int(window.width),
                win_ysize=int(window.height),
            )

        if arr is None:
            raise RuntimeError(f"ReadAsArray() returned None for band {idx}")
        return np.asarray(arr)

    def block_windows(self, bidx: int):
        band = self._ds.GetRasterBand(int(bidx))
        if band is None:
            raise ValueError(f"Invalid band index: {bidx}")
        block_x, block_y = band.GetBlockSize()
        if block_x <= 0 or block_y <= 0:
            block_x, block_y = 256, 256

        n_cols = (self.width + block_x - 1) // block_x
        n_rows = (self.height + block_y - 1) // block_y
        for r in range(n_rows):
            row_off = r * block_y
            h = min(block_y, self.height - row_off)
            for c in range(n_cols):
                col_off = c * block_x
                w = min(block_x, self.width - col_off)
                yield (r, c), Window(col_off=col_off, row_off=row_off, width=w, height=h)


@contextmanager
def open_geotiff(path: str):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open: {path}")
    try:
        yield GeoDataset(ds)
    finally:
        # release underlying dataset handle
        ds = None


def detect_layout(src: GeoDataset, fast: bool) -> Layout:
    count = src.count
    # infer (day_count, per_day, static_count)
    # default static_count=4
    static_count = 4
    dyn = count - static_count
    if dyn <= 0:
        raise ValueError(f"band_count={count} < static_count={static_count}")

    candidates: List[Layout] = []
    if dyn % 2 == 0:
        day_count = dyn // 2
        if day_count == 181:
            candidates += [
                Layout2InterleavedNDSI_MASK(
                    "2x_interleaved_NDSI_MASK", 2, day_count, static_count
                ),
                Layout2InterleavedMASK_NDSI(
                    "2x_interleaved_MASK_NDSI", 2, day_count, static_count
                ),
                Layout2GroupedNDSI_then_MASK(
                    "2x_grouped_NDSI_then_MASK", 2, day_count, static_count
                ),
                Layout2GroupedMASK_then_NDSI(
                    "2x_grouped_MASK_then_NDSI", 2, day_count, static_count
                ),
            ]
    if dyn % 3 == 0:
        day_count = dyn // 3
        if day_count == 181:
            candidates += [
                Layout3InterleavedNDSI_QMASK_MASK(
                    "3x_interleaved_NDSI_QMASK_MASK", 3, day_count, static_count
                ),
                Layout3InterleavedNDSI_MASK_QMASK(
                    "3x_interleaved_NDSI_MASK_QMASK", 3, day_count, static_count
                ),
                Layout3InterleavedMASK_NDSI_QMASK(
                    "3x_interleaved_MASK_NDSI_QMASK", 3, day_count, static_count
                ),
                Layout3InterleavedMASK_QMASK_NDSI(
                    "3x_interleaved_MASK_QMASK_NDSI", 3, day_count, static_count
                ),
                Layout3GroupedNDSI_QMASK_MASK(
                    "3x_grouped_NDSI_QMASK_MASK", 3, day_count, static_count
                ),
                Layout3GroupedNDSI_MASK_QMASK(
                    "3x_grouped_NDSI_MASK_QMASK", 3, day_count, static_count
                ),
            ]

    if not candidates:
        raise ValueError(
            f"Cannot infer layout from band_count={count} (dyn={dyn}, static=4)."
        )

    # score each layout on sample days
    if fast:
        sample_ts = [0, 90, 180]
    else:
        # still just detect on 3 days; full scan later uses this layout
        sample_ts = [0, 90, 180]

    best = None
    for lay in candidates:
        mism_list = []
        pseudo_list = []
        masklike_list = []
        ndsilike_list = []
        qmasklike_list = []
        for t in sample_ts:
            ndsi_i, mask_i, qmask_i = lay.band_of_day(t)
            ndsi = src.read(ndsi_i)
            mask = src.read(mask_i)

            # compute mismatch/pseudo0 on full tile for those days
            mismatch = np.mean((mask == 0) != (ndsi == NDSI_FILL))
            pseudo0 = np.mean((mask == 1) & (ndsi == NDSI_FILL))

            mism_list.append(float(mismatch))
            pseudo_list.append(float(pseudo0))
            masklike_list.append(1.0 if is_binary(mask) else 0.0)
            ndsilike_list.append(1.0 if looks_like_ndsi(ndsi) else 0.0)

            if qmask_i is not None:
                qmask = src.read(qmask_i)
                qmasklike_list.append(1.0 if looks_like_qmask(qmask) else 0.0)
            else:
                qmasklike_list.append(1.0)  # no qmask required for 2-band case

        # Score: prioritize hard constraints, then type-likeness
        score = (
            max(mism_list),
            max(pseudo_list),
            1.0 - (sum(masklike_list) / len(masklike_list)),
            1.0 - (sum(ndsilike_list) / len(ndsilike_list)),
            1.0 - (sum(qmasklike_list) / len(qmasklike_list)),
        )
        if best is None or score < best[0]:
            best = (score, lay, mism_list, pseudo_list, masklike_list, ndsilike_list)

    assert best is not None
    return best[1]


# ----------------------------
# Streaming stats utils
# ----------------------------
@dataclass
class Counts:
    total: int = 0
    ndsi_in_0_100: int = 0
    ndsi_eq_255: int = 0
    ndsi_ge_200: int = 0
    mask_0: int = 0
    mask_1: int = 0
    mask_other: int = 0
    mismatch: int = 0
    pseudo0: int = 0
    q0: int = 0
    q1: int = 0
    q2: int = 0
    q3: int = 0
    q_other: int = 0


def update_counts(
    cnt: Counts, ndsi: np.ndarray, mask: np.ndarray, qmask: Optional[np.ndarray] = None
):
    # assume ndsi and mask are same shape
    n = ndsi.size
    cnt.total += n

    cnt.ndsi_in_0_100 += int(np.sum((ndsi >= 0) & (ndsi <= 100)))
    cnt.ndsi_eq_255 += int(np.sum(ndsi == NDSI_FILL))
    cnt.ndsi_ge_200 += int(np.sum(ndsi >= 200))

    cnt.mask_0 += int(np.sum(mask == 0))
    cnt.mask_1 += int(np.sum(mask == 1))
    cnt.mask_other += int(np.sum((mask != 0) & (mask != 1)))

    cnt.mismatch += int(np.sum((mask == 0) != (ndsi == NDSI_FILL)))
    cnt.pseudo0 += int(np.sum((mask == 1) & (ndsi == NDSI_FILL)))

    if qmask is not None:
        cnt.q0 += int(np.sum(qmask == 0))
        cnt.q1 += int(np.sum(qmask == 1))
        cnt.q2 += int(np.sum(qmask == 2))
        cnt.q3 += int(np.sum(qmask == 3))
        cnt.q_other += int(np.sum(~np.isin(qmask, [0, 1, 2, 3])))


@dataclass
class StaticStats:
    elev_eq0: int = 0
    elev_eqfill: int = 0
    slope_eq0: int = 0
    slope_eqfill: int = 0
    north_eq0: int = 0
    north_eqfill: int = 0
    ndvi_eq0: int = 0
    ndvi_eqfill: int = 0  # in case you use 255 or others; usually 0 here
    ndvi_eqfill9999: int = 0
    total: int = 0
    elev_min: Optional[int] = None
    elev_max: Optional[int] = None
    slope_min: Optional[int] = None
    slope_max: Optional[int] = None
    north_min: Optional[int] = None
    north_max: Optional[int] = None
    ndvi_min: Optional[int] = None
    ndvi_max: Optional[int] = None


def _update_minmax(
    cur_min: Optional[int], cur_max: Optional[int], arr: np.ndarray
) -> Tuple[int, int]:
    mn = int(arr.min())
    mx = int(arr.max())
    if cur_min is None:
        return mn, mx
    return min(cur_min, mn), max(cur_max, mx)


def update_static_stats(
    ss: StaticStats,
    elev: np.ndarray,
    slope: np.ndarray,
    north: np.ndarray,
    ndvi: np.ndarray,
):
    n = elev.size
    ss.total += n

    ss.elev_eq0 += int(np.sum(elev == 0))
    ss.elev_eqfill += int(np.sum(elev == STATIC_FILL))

    ss.slope_eq0 += int(np.sum(slope == 0))
    ss.slope_eqfill += int(np.sum(slope == STATIC_FILL))

    ss.north_eq0 += int(np.sum(north == 0))
    ss.north_eqfill += int(np.sum(north == STATIC_FILL))

    ss.ndvi_eq0 += int(np.sum(ndvi == 0))
    ss.ndvi_eqfill9999 += int(np.sum(ndvi == STATIC_FILL))

    ss.elev_min, ss.elev_max = _update_minmax(ss.elev_min, ss.elev_max, elev)
    ss.slope_min, ss.slope_max = _update_minmax(ss.slope_min, ss.slope_max, slope)
    ss.north_min, ss.north_max = _update_minmax(ss.north_min, ss.north_max, north)
    ss.ndvi_min, ss.ndvi_max = _update_minmax(ss.ndvi_min, ss.ndvi_max, ndvi)


# ----------------------------
# ValidDays histogram (0..181)
# ----------------------------
@dataclass
class ValidDaysHist:
    counts: np.ndarray  # length 182

    @staticmethod
    def new():
        return ValidDaysHist(counts=np.zeros(182, dtype=np.int64))

    def add_window(self, sum_mask: np.ndarray):
        h = np.bincount(sum_mask.ravel().astype(np.int32), minlength=182)
        self.counts[:182] += h[:182]

    def mean(self) -> float:
        total = self.counts.sum()
        if total == 0:
            return float("nan")
        values = np.arange(182, dtype=np.float64)
        return float((values * self.counts).sum() / total)

    def percentile(self, p: float) -> float:
        # p in [0,100]
        total = self.counts.sum()
        if total == 0:
            return float("nan")
        target = total * (p / 100.0)
        c = np.cumsum(self.counts)
        idx = int(np.searchsorted(c, target, side="left"))
        return float(idx)


# ----------------------------
# Core checks per tif
# ----------------------------
def check_tif(path: str, fast: bool) -> Dict:
    gid = group_id_from_path(path) or "UNKNOWN_GROUP"
    yrs = years_from_group_id(gid)
    start_year = yrs[0] if yrs else None
    end_year = yrs[1] if yrs else None

    with open_geotiff(path) as src:
        xsize, ysize = src.width, src.height
        band_count = src.count

        layout = detect_layout(src, fast=fast)
        day_count = layout.day_count
        per_day = layout.per_day
        expected_band_count = day_count * per_day + 4
        expected_ok = band_count == expected_band_count

        # expected dates from group id years (strict)
        if start_year is not None:
            dates = winter_dates(start_year)
            expected_start = dates[0]
            expected_end = dates[-1]
            feb29 = has_feb29_in_season(start_year)
        else:
            dates = None
            expected_start = None
            expected_end = None
            feb29 = None

        # In exported tif (no band names guaranteed), we validate by structure:
        ndsi_dates_ok = day_count == 181
        ndsi_start_ok = True if expected_start else None
        ndsi_end_ok = True if expected_end else None
        ndsi_missing_dates_count = 0
        ndsi_extra_dates_count = 0

        # Dynamic full scan (stream blocks)
        cnt = Counts()
        # ValidDays histogram (per pixel sum of mask across all days)
        vhist = ValidDaysHist.new()

        # Static stats
        ss = StaticStats()

        # Window iteration
        # In fast mode, we only scan 3 days and a subset of windows
        if fast:
            day_indices = [0, 90, 180]
            # take first N windows
            win_iter = list(src.block_windows(1))[:50]
        else:
            day_indices = list(range(day_count))
            win_iter = list(src.block_windows(1))

        # For validDays histogram we need ALL days; in fast mode we skip it (still provide NA)
        compute_valid_days = not fast

        # Precompute static band indices: last 4
        elev_i = band_count - 3
        slope_i = band_count - 2
        north_i = band_count - 1
        ndvi_i = band_count

        # Process per window
        for _, win in win_iter:
            # Static window read
            elev = src.read(elev_i, window=win)
            slope = src.read(slope_i, window=win)
            north = src.read(north_i, window=win)
            ndvi = src.read(ndvi_i, window=win)
            update_static_stats(ss, elev, slope, north, ndvi)

            # Dynamic counts: loop day indices
            if not fast:
                # For full scan, read mask stack window once for validDays
                if compute_valid_days:
                    # build list of mask band indexes across days
                    mask_band_indexes = []
                    qmask_band_indexes = []
                    for t in range(day_count):
                        ndsi_b, mask_b, qmask_b = layout.band_of_day(t)
                        mask_band_indexes.append(mask_b)
                        if qmask_b is not None:
                            qmask_band_indexes.append(qmask_b)

                    # read all mask bands for this window: shape (D, h, w)
                    mask_stack = src.read(mask_band_indexes, window=win)
                    # sum across days
                    sum_mask = np.sum(mask_stack == 1, axis=0).astype(np.int32)
                    vhist.add_window(sum_mask)

            # Dynamic stats on chosen days (fast: 3 days, full: all days)
            for t in day_indices:
                ndsi_b, mask_b, qmask_b = layout.band_of_day(t)
                ndsi = src.read(ndsi_b, window=win)
                mask = src.read(mask_b, window=win)
                qmask = src.read(qmask_b, window=win) if qmask_b is not None else None
                update_counts(cnt, ndsi, mask, qmask=qmask)

        # ratios
        total = max(cnt.total, 1)
        out = {
            "tif_path": path.replace("\\", "/"),
            "group_id": gid,
            "xsize": xsize,
            "ysize": ysize,
            "band_count": band_count,
            "expected_band_count": expected_band_count,
            "expected_band_count_ok": expected_ok,
            "layout": layout.name,
            "per_day": per_day,
            "day_count": day_count,
            "ndsi_dates_ok": ndsi_dates_ok,
            "ndsi_has_feb29": bool(feb29) if feb29 is not None else None,
            "ndsi_start_ymd": expected_start,
            "ndsi_end_ymd": expected_end,
            "ndsi_start_ok": ndsi_start_ok,
            "ndsi_end_ok": ndsi_end_ok,
            "ndsi_actual_len": day_count,
            "ndsi_missing_dates_count": ndsi_missing_dates_count,
            "ndsi_extra_dates_count": ndsi_extra_dates_count,
            "ndsi_ratio_in_0_100": cnt.ndsi_in_0_100 / total,
            "ndsi_ratio_eq_255": cnt.ndsi_eq_255 / total,
            "ndsi_ratio_ge_200": cnt.ndsi_ge_200 / total,
            "mask_ratio_0": cnt.mask_0 / total,
            "mask_ratio_1": cnt.mask_1 / total,
            "mask_ratio_other": cnt.mask_other / total,
            "qmask_ratio_0": cnt.q0 / total if per_day == 3 else None,
            "qmask_ratio_1": cnt.q1 / total if per_day == 3 else None,
            "qmask_ratio_2": cnt.q2 / total if per_day == 3 else None,
            "qmask_ratio_3": cnt.q3 / total if per_day == 3 else None,
            "qmask_ratio_other": cnt.q_other / total if per_day == 3 else None,
            "qmask_mask_mismatch_ratio": (
                cnt.mismatch / total
            ),  # name kept for compatibility
            "mismatch_ratio": cnt.mismatch / total,
            "pseudo0_ratio": cnt.pseudo0 / total,
            "static_elev_ratio_eq_0": ss.elev_eq0 / max(ss.total, 1),
            "static_elev_ratio_eq_-9999": ss.elev_eqfill / max(ss.total, 1),
            "static_slope_ratio_eq_0": ss.slope_eq0 / max(ss.total, 1),
            "static_slope_ratio_eq_-9999": ss.slope_eqfill / max(ss.total, 1),
            "static_north_ratio_eq_0": ss.north_eq0 / max(ss.total, 1),
            "static_north_ratio_eq_-9999": ss.north_eqfill / max(ss.total, 1),
            "static_ndvi_ratio_eq_0": ss.ndvi_eq0 / max(ss.total, 1),
            "static_ndvi_ratio_eq_-9999": ss.ndvi_eqfill9999 / max(ss.total, 1),
            "elev_min": ss.elev_min,
            "elev_max": ss.elev_max,
            "slope_min": ss.slope_min,
            "slope_max": ss.slope_max,
            "north_min": ss.north_min,
            "north_max": ss.north_max,
            "ndvi_min": ss.ndvi_min,
            "ndvi_max": ss.ndvi_max,
            "validDays_mean": vhist.mean() if not fast else None,
            "validDays_p5": vhist.percentile(5) if not fast else None,
            "validDays_p25": vhist.percentile(25) if not fast else None,
            "validDays_p50": vhist.percentile(50) if not fast else None,
            "validDays_p75": vhist.percentile(75) if not fast else None,
            "validDays_p95": vhist.percentile(95) if not fast else None,
        }
        return out


def weighted_merge(rows: List[Dict], keys: List[str], weight_key: str) -> Dict:
    # Weighted average for ratios; min/max for extrema; boolean AND for ok flags
    wsum = 0.0
    out = {}
    for r in rows:
        w = float(r[weight_key])
        wsum += w
        for k in keys:
            v = r.get(k, None)
            if v is None:
                continue
            out[k] = out.get(k, 0.0) + w * float(v)
    if wsum > 0:
        for k in list(out.keys()):
            out[k] /= wsum
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="Path like data/raw_data/QinghaiTibet"
    )
    ap.add_argument(
        "--only", default=None, help="Substring filter for group_id / filename"
    )
    ap.add_argument(
        "--fast", action="store_true", help="Fast smoke test (3 days, limited windows)"
    )
    args = ap.parse_args()

    root = args.root
    only = args.only
    fast = args.fast

    root_path = os.path.normpath(root)
    if not os.path.exists(root_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        candidate = os.path.normpath(os.path.join(repo_root, root_path))
        if os.path.exists(candidate):
            root_path = candidate
            root = candidate

    tifs = []
    for dp, _, files in os.walk(root_path):
        for fn in files:
            if fn.lower().endswith(".tif"):
                p = os.path.join(dp, fn)
                if only and (only not in fn):
                    continue
                tifs.append(p)
    tifs.sort()

    if not tifs:
        raise SystemExit(f"No .tif found under {root}")

    # per-tile rows
    tile_rows: List[Dict] = []
    errors = 0

    print("=" * 88)
    print(f"[QC] root={root}  tiles={len(tifs)}  mode={'FAST' if fast else 'FULL'}")
    print("=" * 88)

    for i, p in enumerate(tifs, 1):
        try:
            row = check_tif(p, fast=fast)
            tile_rows.append(row)

            # Hard redlines
            mismatch = row["mismatch_ratio"]
            pseudo0 = row["pseudo0_ratio"]
            band_ok = row["expected_band_count_ok"]
            dates_ok = row["ndsi_dates_ok"]

            status = "PASS"
            if (
                (not band_ok)
                or (not dates_ok)
                or (mismatch > 0)
                or (pseudo0 > 0)
                or (row["mask_ratio_other"] > 0)
            ):
                status = "FAIL"

            print(f"[{status}] {i:04d}/{len(tifs):04d}  {os.path.basename(p)}")
            print(f"  group_id={row['group_id']}")
            print(
                f"  layout={row['layout']}  bands={row['band_count']} (expect {row['expected_band_count']})  day_count={row['day_count']} per_day={row['per_day']}"
            )
            print(
                f"  dates_ok={dates_ok}  feb29_in_season={row['ndsi_has_feb29']}  start={row['ndsi_start_ymd']} end={row['ndsi_end_ymd']}"
            )
            print(
                f"  ndsi_in0_100={row['ndsi_ratio_in_0_100']:.6f}  ndsi_eq255={row['ndsi_ratio_eq_255']:.6f}  ndsi_ge200={row['ndsi_ratio_ge_200']:.6f}"
            )
            print(
                f"  mask0={row['mask_ratio_0']:.6f}  mask1={row['mask_ratio_1']:.6f}  mask_other={row['mask_ratio_other']:.6f}"
            )
            if row["qmask_ratio_0"] is not None:
                print(
                    f"  qmask[0/1/2/3/other]={row['qmask_ratio_0']:.6f}/{row['qmask_ratio_1']:.6f}/{row['qmask_ratio_2']:.6f}/{row['qmask_ratio_3']:.6f}/{row['qmask_ratio_other']:.6f}"
                )
            print(f"  mismatch={mismatch:.9f}  pseudo0={pseudo0:.9f}")
            print(
                f"  static elev(0/fill)={row['static_elev_ratio_eq_0']:.6f}/{row['static_elev_ratio_eq_-9999']:.6f}  "
                f"slope(0/fill)={row['static_slope_ratio_eq_0']:.6f}/{row['static_slope_ratio_eq_-9999']:.6f}  "
                f"north(0/fill)={row['static_north_ratio_eq_0']:.6f}/{row['static_north_ratio_eq_-9999']:.6f}  "
                f"ndvi(0/fill)={row['static_ndvi_ratio_eq_0']:.6f}/{row['static_ndvi_ratio_eq_-9999']:.6f}"
            )
            print(
                f"  static min/max: elev={row['elev_min']}/{row['elev_max']}  slope={row['slope_min']}/{row['slope_max']}  "
                f"north={row['north_min']}/{row['north_max']}  ndvi={row['ndvi_min']}/{row['ndvi_max']}"
            )
            if not fast:
                print(
                    f"  validDays mean/p5/p25/p50/p75/p95 = "
                    f"{row['validDays_mean']:.3f}/{row['validDays_p5']:.1f}/{row['validDays_p25']:.1f}/{row['validDays_p50']:.1f}/{row['validDays_p75']:.1f}/{row['validDays_p95']:.1f}"
                )

            # Extra invariant checks (prints as WARN, doesn't change PASS/FAIL)
            if abs(row["mask_ratio_0"] - row["ndsi_ratio_eq_255"]) > 1e-9:
                print("  [WARN] mask0 != ndsi_eq255 (should be identical if derived).")
            if abs(row["ndsi_ratio_ge_200"] - row["ndsi_ratio_eq_255"]) > 1e-9:
                print(
                    "  [WARN] ndsi_ge200 != ndsi_eq255 (after cleaning they should match)."
                )
            if row["static_elev_ratio_eq_-9999"] > 0:
                print(
                    "  [WARN] Elevation has STATIC_FILL pixels (possible DEM void or reprojection hole)."
                )
            if row["static_ndvi_ratio_eq_-9999"] > 0:
                print(
                    "  [WARN] NDVI has STATIC_FILL pixels (time window / ROI / masking issue)."
                )

            print("-" * 88)

        except Exception as e:
            errors += 1
            print(f"[ERROR] {i:04d}/{len(tifs):04d}  {p}")
            print(f"  {type(e).__name__}: {e}")
            print("-" * 88)

    # write tile csv
    os.makedirs("qc_out", exist_ok=True)
    tiles_csv = os.path.join("qc_out", "qc_tiles.csv")
    groups_csv = os.path.join("qc_out", "qc_groups.csv")
    report_txt = os.path.join("qc_out", "qc_report.txt")

    # Save console report
    # (simple: re-run output not captured; so we just write summary + worst tiles)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(
            f"[QC] root={root} tiles={len(tifs)} mode={'FAST' if fast else 'FULL'} errors={errors}\n"
        )

    # Write CSVs
    if tile_rows:
        # Stable header
        headers = list(tile_rows[0].keys())
        with open(tiles_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in tile_rows:
                w.writerow(r)

        # Group aggregation
        by_gid: Dict[str, List[Dict]] = {}
        for r in tile_rows:
            by_gid.setdefault(r["group_id"], []).append(r)

        group_rows = []
        ratio_keys = [
            "ndsi_ratio_in_0_100",
            "ndsi_ratio_eq_255",
            "ndsi_ratio_ge_200",
            "mask_ratio_0",
            "mask_ratio_1",
            "mask_ratio_other",
            "qmask_ratio_0",
            "qmask_ratio_1",
            "qmask_ratio_2",
            "qmask_ratio_3",
            "qmask_ratio_other",
            "mismatch_ratio",
            "pseudo0_ratio",
            "static_elev_ratio_eq_0",
            "static_elev_ratio_eq_-9999",
            "static_slope_ratio_eq_0",
            "static_slope_ratio_eq_-9999",
            "static_north_ratio_eq_0",
            "static_north_ratio_eq_-9999",
            "static_ndvi_ratio_eq_0",
            "static_ndvi_ratio_eq_-9999",
        ]
        # Use pixel count as weight
        for gid, rows in by_gid.items():
            for r in rows:
                r["_pixels"] = int(r["xsize"]) * int(r["ysize"])
            merged = weighted_merge(
                rows, [k for k in ratio_keys if rows[0].get(k) is not None], "_pixels"
            )

            # Min/max extrema across tiles
            elev_min = min(
                [r["elev_min"] for r in rows if r["elev_min"] is not None], default=None
            )
            elev_max = max(
                [r["elev_max"] for r in rows if r["elev_max"] is not None], default=None
            )
            slope_min = min(
                [r["slope_min"] for r in rows if r["slope_min"] is not None],
                default=None,
            )
            slope_max = max(
                [r["slope_max"] for r in rows if r["slope_max"] is not None],
                default=None,
            )
            north_min = min(
                [r["north_min"] for r in rows if r["north_min"] is not None],
                default=None,
            )
            north_max = max(
                [r["north_max"] for r in rows if r["north_max"] is not None],
                default=None,
            )
            ndvi_min = min(
                [r["ndvi_min"] for r in rows if r["ndvi_min"] is not None], default=None
            )
            ndvi_max = max(
                [r["ndvi_max"] for r in rows if r["ndvi_max"] is not None], default=None
            )

            # ValidDays aggregation (hist already per tile, but we only computed per tile, so use pixel-weight mean here)
            if not fast:
                vmean = weighted_merge(
                    rows,
                    [
                        "validDays_mean",
                        "validDays_p5",
                        "validDays_p25",
                        "validDays_p50",
                        "validDays_p75",
                        "validDays_p95",
                    ],
                    "_pixels",
                )
            else:
                vmean = {}

            group_row = {
                "group_id": gid,
                "tiles": len(rows),
                "band_count_all_ok": all(r["expected_band_count_ok"] for r in rows),
                "ndsi_dates_all_ok": all(r["ndsi_dates_ok"] for r in rows),
                "ndsi_any_feb29": any(
                    bool(r["ndsi_has_feb29"])
                    for r in rows
                    if r["ndsi_has_feb29"] is not None
                ),
                "ndsi_start_ymd": rows[0]["ndsi_start_ymd"],
                "ndsi_end_ymd": rows[0]["ndsi_end_ymd"],
                "layout_set": ",".join(sorted(set(r["layout"] for r in rows))),
                "elev_min": elev_min,
                "elev_max": elev_max,
                "slope_min": slope_min,
                "slope_max": slope_max,
                "north_min": north_min,
                "north_max": north_max,
                "ndvi_min": ndvi_min,
                "ndvi_max": ndvi_max,
            }
            group_row.update(merged)
            group_row.update(vmean)
            group_rows.append(group_row)

        # write groups csv
        headers_g = list(group_rows[0].keys()) if group_rows else []
        with open(groups_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers_g)
            w.writeheader()
            for r in group_rows:
                w.writerow(r)

        # Print final summary + worst tiles
        worst = sorted(
            tile_rows,
            key=lambda r: (-(r["mismatch_ratio"] or 0.0), -(r["pseudo0_ratio"] or 0.0)),
        )[:10]
        print("=" * 88)
        print(f"[QC DONE] tiles={len(tile_rows)} errors={errors}")
        print(f"Outputs: {tiles_csv} , {groups_csv} , {report_txt}")
        print("[Worst tiles by mismatch/pseudo0]")
        for w in worst:
            print(
                f"  mismatch={w['mismatch_ratio']:.9f} pseudo0={w['pseudo0_ratio']:.9f}  {w['tif_path']}"
            )
        print("=" * 88)
    else:
        print("[QC DONE] No valid tile rows produced (all errors).")


if __name__ == "__main__":
    main()
