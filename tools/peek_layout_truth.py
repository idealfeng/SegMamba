import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from osgeo import gdal

NDSI_FILL = 255


def winter_dates(start_year: int):
    d0 = date(start_year, 11, 1)
    d1 = date(start_year + 1, 4, 30)
    out = []
    cur = d0
    while cur <= d1:
        if not (cur.month == 2 and cur.day == 29):
            out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


GROUP_RE = re.compile(
    r"^(?P<gid>.+?_\d{4}_\d{4}_.+?)(?:-\d{10}-\d{10})?\.tif$", re.IGNORECASE
)


def group_id_from_path(p: str):
    base = os.path.basename(p)
    m = GROUP_RE.match(base)
    return m.group("gid") if m else "UNKNOWN_GROUP"


def years_from_gid(gid: str):
    m = re.search(r"_(\d{4})_(\d{4})_", gid)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True)
class Window:
    col_off: int
    row_off: int
    width: int
    height: int


def iter_windows(ds: gdal.Dataset, bidx: int, *, max_windows: int) -> list[Window]:
    band = ds.GetRasterBand(int(bidx))
    if band is None:
        return []
    block_x, block_y = band.GetBlockSize()
    if block_x <= 0 or block_y <= 0:
        block_x, block_y = 256, 256

    w = int(ds.RasterXSize)
    h = int(ds.RasterYSize)
    n_cols = (w + block_x - 1) // block_x
    n_rows = (h + block_y - 1) // block_y

    out: list[Window] = []
    for r in range(n_rows):
        row_off = r * block_y
        hh = min(block_y, h - row_off)
        for c in range(n_cols):
            col_off = c * block_x
            ww = min(block_x, w - col_off)
            out.append(Window(col_off=col_off, row_off=row_off, width=ww, height=hh))
            if len(out) >= max_windows:
                return out
    return out


def read_window(ds: gdal.Dataset, bidx: int, win: Window) -> np.ndarray:
    band = ds.GetRasterBand(int(bidx))
    if band is None:
        raise ValueError(f"Invalid band index: {bidx}")
    arr = band.ReadAsArray(
        xoff=int(win.col_off),
        yoff=int(win.row_off),
        win_xsize=int(win.width),
        win_ysize=int(win.height),
    )
    if arr is None:
        raise RuntimeError(f"ReadAsArray() returned None for band {bidx}")
    return np.asarray(arr)


def mismatch_stats_sampled(
    ds: gdal.Dataset,
    ndsi_b: int,
    mask_b: int,
    *,
    max_windows: int,
) -> tuple[float, float, float, float, bool, bool]:
    wins = iter_windows(ds, 1, max_windows=max_windows)
    if not wins:
        wins = [
            Window(
                col_off=0,
                row_off=0,
                width=int(ds.RasterXSize),
                height=int(ds.RasterYSize),
            )
        ]

    total = 0
    mismatch = 0
    pseudo0 = 0
    mask0 = 0
    ndsi255 = 0
    mask_unique: set[int] = set()
    ndsi_min = None
    ndsi_max = None

    for win in wins:
        ndsi = read_window(ds, ndsi_b, win)
        mask = read_window(ds, mask_b, win)
        total += int(ndsi.size)

        mismatch += int(np.sum((mask == 0) != (ndsi == NDSI_FILL)))
        pseudo0 += int(np.sum((mask == 1) & (ndsi == NDSI_FILL)))
        mask0 += int(np.sum(mask == 0))
        ndsi255 += int(np.sum(ndsi == NDSI_FILL))

        # sample uniques cheaply
        mu = np.unique(
            mask[:: max(1, mask.shape[0] // 32), :: max(1, mask.shape[1] // 32)]
        )
        for v in mu.tolist()[:20]:
            try:
                mask_unique.add(int(v))
            except Exception:
                pass

        mn = int(ndsi.min())
        mx = int(ndsi.max())
        ndsi_min = mn if ndsi_min is None else min(ndsi_min, mn)
        ndsi_max = mx if ndsi_max is None else max(ndsi_max, mx)

    denom = max(total, 1)
    mask_like = len(mask_unique) <= 3 and mask_unique.issubset({0, 1})
    ndsi_like = (
        ndsi_min is not None
        and ndsi_max is not None
        and ndsi_min >= 0
        and ndsi_max <= 255
        and (ndsi_max == 255 or (ndsi255 / denom) > 1e-4)
    )

    return (
        mismatch / denom,
        pseudo0 / denom,
        mask0 / denom,
        ndsi255 / denom,
        mask_like,
        ndsi_like,
    )


def resolve_root(root: str | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if root:
        p = Path(root)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    candidates = [
        repo_root / "data" / "raw_data" / "Tibet",
        repo_root / "data" / "raw_data" / "QinghaiTibet",
        repo_root / "data" / "raw_data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def iter_tifs(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".tif":
        return [root]
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.tif"))


def peek_one(path: str, day_index: int, *, max_windows: int):
    gid = group_id_from_path(path)
    yrs = years_from_gid(gid)
    day = winter_dates(yrs[0])[day_index] if yrs else f"idx{day_index}"

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open: {path}")

    D = 181
    total = int(ds.RasterCount)
    static_count = 4
    dyn = total - static_count

    layouts = {
        "2x_interleaved_NDSI_MASK": lambda t: (2 * t + 1, 2 * t + 2),
        "2x_interleaved_MASK_NDSI": lambda t: (2 * t + 2, 2 * t + 1),
        "2x_grouped_NDSI_then_MASK": lambda t: (t + 1, D + t + 1),
        "2x_grouped_MASK_then_NDSI": lambda t: (D + t + 1, t + 1),
    }

    results = []
    for name, fn in layouts.items():
        ndsi_i, mask_i = fn(day_index)
        if ndsi_i < 1 or mask_i < 1 or ndsi_i > total or mask_i > total:
            continue
        mismatch, pseudo0, mask0, ndsi255, mask_like, ndsi_like = mismatch_stats_sampled(
            ds, ndsi_i, mask_i, max_windows=max_windows
        )
        results.append(
            {
                "mismatch": mismatch,
                "pseudo0": pseudo0,
                "layout": name,
                "ndsi_band": ndsi_i,
                "mask_band": mask_i,
                "mask_like": mask_like,
                "ndsi_like": ndsi_like,
                "mask0": mask0,
                "ndsi255": ndsi255,
            }
        )

    results.sort(key=lambda r: (r["mismatch"], r["pseudo0"]))
    best = results[0] if results else None

    return {
        "tif_path": Path(path).as_posix(),
        "group_id": gid,
        "day": day,
        "day_index": day_index,
        "band_count": total,
        "dyn_band_count": dyn,
        "best_layout": (best["layout"] if best else ""),
        "best_ndsi_band": (best["ndsi_band"] if best else ""),
        "best_mask_band": (best["mask_band"] if best else ""),
        "best_mismatch": (best["mismatch"] if best else ""),
        "best_pseudo0": (best["pseudo0"] if best else ""),
        "best_mask0": (best["mask0"] if best else ""),
        "best_ndsi255": (best["ndsi255"] if best else ""),
        "best_mask_like": (best["mask_like"] if best else ""),
        "best_ndsi_like": (best["ndsi_like"] if best else ""),
    }


def main():
    gdal.UseExceptions()

    ap = argparse.ArgumentParser(
        description=(
            "Scan raw_data GeoTIFFs and infer the most plausible 2-band layout "
            "by checking MASK/NDSI fill alignment on a few days (sampled windows; no rasterio)."
        )
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Folder containing .tif files (default: auto-detect data/raw_data/Tibet -> QinghaiTibet).",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: tools/qc_out/peek_layout_truth.csv).",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=50,
        help="Max block windows to sample per band (default: 50).",
    )
    ap.add_argument(
        "--days",
        default="0,90,180",
        help="Comma-separated day indices to test (default: 0,90,180).",
    )
    args = ap.parse_args()

    root = resolve_root(args.root)
    tifs = iter_tifs(root)
    if not tifs:
        raise SystemExit(f"No .tif found under: {root}")

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "qc_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "peek_layout_truth.csv")
    if not out_csv.is_absolute():
        out_csv = (Path.cwd() / out_csv).resolve()

    day_indices = []
    for part in str(args.days).split(","):
        part = part.strip()
        if part == "":
            continue
        day_indices.append(int(part))

    rows = []
    print(f"[PEEK] root={root} tifs={len(tifs)} out={out_csv}")
    for i, tif in enumerate(tifs, 1):
        tif_rows = []
        for di in day_indices:
            row = peek_one(str(tif), day_index=di, max_windows=int(args.max_windows))
            tif_rows.append(row)
            rows.append(row)
        best_layouts = sorted({r["best_layout"] for r in tif_rows if r["best_layout"]})
        print(
            f"[{i:04d}/{len(tifs):04d}] {tif.name}  best={','.join(best_layouts) if best_layouts else 'N/A'}"
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[PEEK DONE] wrote {out_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
