# validate_era5_geotiff.py评价气象数据
# Validate ERA5-Land winter stacks exported from GEE:
# - expect 181 bands (day_000..day_180 or b1..b181)
# - basic range / fill checks
# - optional: grid alignment check vs a reference MODIS tile

import argparse
import math
import re
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

FILL_VALUE_DEFAULT = -9999.0
ERA5_NATIVE_VAR_RE = re.compile(r"_ERA5L_(?P<var>T2mC|PrcpMM)_", re.IGNORECASE)
ERA5_ALIGNED_VAR_RE = re.compile(r"\.ERA5(?P<var>T2mC|PrcpMM)\.", re.IGNORECASE)


def _open(path: str):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return ds


def _get_gt(ds):
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        return None
    return tuple(float(x) for x in gt)


def _wkt(ds):
    return (ds.GetProjectionRef() or "").strip()


def _band_descs(ds, max_check=6):
    # Try to detect naming pattern from a few bands
    descs = []
    for i in range(1, min(ds.RasterCount, max_check) + 1):
        b = ds.GetRasterBand(i)
        descs.append((i, (b.GetDescription() or "").strip()))
    return descs


def _downsample_band_stats(band, fill_value: float, buf=256):
    # Read as a downsampled array to avoid huge memory
    xsize = band.XSize
    ysize = band.YSize
    buf_x = min(buf, xsize)
    buf_y = min(buf, ysize)

    arr = band.ReadAsArray(0, 0, xsize, ysize, buf_xsize=buf_x, buf_ysize=buf_y)
    if arr is None:
        raise RuntimeError("ReadAsArray returned None")

    arr = arr.astype("float64", copy=False)

    # mask fill (treat NaN/Inf as fill too)
    m = (arr == fill_value) | (~np.isfinite(arr))
    valid = arr[~m]
    fill_ratio = float(m.mean()) if arr.size else 0.0

    if valid.size == 0:
        return {
            "fill_ratio": fill_ratio,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "p01": math.nan,
            "p99": math.nan,
        }

    p01 = float(np.percentile(valid, 1))
    p99 = float(np.percentile(valid, 99))
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    vmean = float(np.mean(valid))

    return {
        "fill_ratio": fill_ratio,
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "p01": p01,
        "p99": p99,
    }


def _check_grid(ref_ds, ds, tol_origin=1e-6, tol_px=1e-9):
    ref_gt = _get_gt(ref_ds)
    gt = _get_gt(ds)
    if ref_gt is None or gt is None:
        return False, "missing geotransform"

    # size must match exactly
    if ref_ds.RasterXSize != ds.RasterXSize or ref_ds.RasterYSize != ds.RasterYSize:
        return (
            False,
            f"size mismatch ref={ref_ds.RasterXSize}x{ref_ds.RasterYSize} vs {ds.RasterXSize}x{ds.RasterYSize}",
        )

    # pixel size match
    if abs(ref_gt[1] - gt[1]) > tol_px or abs(ref_gt[5] - gt[5]) > tol_px:
        return (
            False,
            f"pixel size mismatch ref=({ref_gt[1]},{ref_gt[5]}) vs ({gt[1]},{gt[5]})",
        )

    # origin match
    if abs(ref_gt[0] - gt[0]) > tol_origin or abs(ref_gt[3] - gt[3]) > tol_origin:
        return (
            False,
            f"origin mismatch ref=({ref_gt[0]},{ref_gt[3]}) vs ({gt[0]},{gt[3]})",
        )

    # rotation terms should be ~0
    if abs(gt[2]) > tol_px or abs(gt[4]) > tol_px:
        return False, f"rotation terms not ~0: rot=({gt[2]},{gt[4]})"

    # CRS match (string compare is OK if both come from GDAL)
    ref_wkt = _wkt(ref_ds)
    wkt = _wkt(ds)
    if ref_wkt and wkt and (ref_wkt != wkt):
        return False, "CRS WKT mismatch (ref vs ds)"

    return True, "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default=None,
        help="Validate a single ERA5 GeoTIFF/VRT path (181-band). If omitted, scans --root.",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Folder containing ERA5 exports (default: data/raw_data/QinghaiTibet_ERA5L).",
    )
    ap.add_argument(
        "--var",
        default=None,
        choices=["temp", "prcp"],
        help="Force variable type (temp=Celsius, prcp=mm/day). If omitted, inferred from filename.",
    )
    ap.add_argument("--expect_bands", type=int, default=181)
    ap.add_argument("--fill_value", type=float, default=FILL_VALUE_DEFAULT)
    ap.add_argument(
        "--ref", default=None, help="Optional MODIS tile tif to check grid alignment."
    )
    ap.add_argument(
        "--sample_bands",
        nargs="+",
        default=["1", "91", "181"],
        help=(
            "1-based band indices to sample stats on. "
            "Accepts space-separated values (PowerShell-friendly) or comma-separated tokens."
        ),
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: tools/qc_out/era5_validate.csv).",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Optional substring filter on filename (useful for a single year).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of files to validate (0 = no limit).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    def _default_root() -> Path:
        return repo_root / "data" / "raw_data" / "QinghaiTibet_ERA5L"

    def _resolve_path(p: Path) -> Path:
        if p.is_absolute():
            return p
        return (Path.cwd() / p).resolve()

    def _iter_inputs() -> list[Path]:
        if args.path:
            return [_resolve_path(Path(args.path))]
        root = _resolve_path(Path(args.root)) if args.root else _default_root()
        if not root.exists():
            raise SystemExit(f"Root not found: {root}")
        if root.is_file():
            return [root]
        files = sorted([*root.rglob("*.tif"), *root.rglob("*.vrt")])
        if args.only:
            files = [p for p in files if args.only in p.name]
        if args.limit and args.limit > 0:
            files = files[: int(args.limit)]
        if not files:
            raise SystemExit(f"No .tif/.vrt found under: {root}")
        return files

    def _infer_var(p: Path) -> str | None:
        # native: Tibetan_ERA5L_T2mC_native_2010_2011_181d.tif
        m = ERA5_NATIVE_VAR_RE.search(p.name)
        if m:
            token = m.group("var").lower()
            return "temp" if token == "t2mc" else ("prcp" if token == "prcpmm" else None)

        # aligned: <MODIS_tile_stem>.ERA5T2mC.ALIGN.vrt / .ERA5PrcpMM.ALIGN.vrt
        m = ERA5_ALIGNED_VAR_RE.search(p.name)
        if m:
            token = m.group("var").lower()
            return "temp" if token == "t2mc" else ("prcp" if token == "prcpmm" else None)

        # fallback: substring match (robust to minor naming tweaks)
        up = p.name.upper()
        if "ERA5T2MC" in up:
            return "temp"
        if "ERA5PRCPMM" in up:
            return "prcp"
        return None

    def _ranges_for(var: str) -> tuple[float, float]:
        if var == "temp":
            return -90.0, 60.0
        return -1e-6, 500.0

    # output CSV default
    out_csv = Path(args.out_csv) if args.out_csv else (script_dir / "qc_out" / "era5_validate.csv")
    if not out_csv.is_absolute():
        out_csv = (Path.cwd() / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    paths = _iter_inputs()

    ref_ds = _open(args.ref) if args.ref else None

    for i, p in enumerate(paths, 1):
        var = args.var or _infer_var(p)
        if var is None:
            print(f"[SKIP] cannot infer --var from filename: {p.name}")
            continue

        print("=" * 88)
        print("[ERA5 VALIDATE]")
        print(f"  [{i:04d}/{len(paths):04d}] path={p}")
        print(f"  var={var}  expect_bands={args.expect_bands}  fill_value={args.fill_value}")
        print("=" * 88)

        ds = _open(str(p))
        bands = ds.RasterCount

        # band descriptions quick peek
        descs = _band_descs(ds)

        grid_ok = None
        grid_msg = None
        if ref_ds is not None:
            ok, msg = _check_grid(ref_ds, ds)
            grid_ok = bool(ok)
            grid_msg = msg
            print(f"[GRID] {msg}")

        # sample bands
        try:
            parts = []
            for token in args.sample_bands:
                parts += [p.strip() for p in str(token).split(",")]
            idxs = [int(x) for x in parts if x]
            idxs = [j for j in idxs if 1 <= j <= max(1, bands)]
            if not idxs:
                idxs = [1, min(91, bands), bands]
        except Exception:
            idxs = [1, min(91, bands), bands]

        hard_min, hard_max = _ranges_for(var)
        any_range_fail = False
        sampled = []
        for bi in idxs:
            band = ds.GetRasterBand(bi)
            st = _downsample_band_stats(band, args.fill_value, buf=256)
            sampled.append((bi, st))
            if not (math.isnan(st["p01"]) or math.isnan(st["p99"])):
                if st["p01"] < hard_min or st["p99"] > hard_max:
                    any_range_fail = True

        if bands != args.expect_bands:
            print(f"[FAIL] band_count={bands} (expect {args.expect_bands})")
        else:
            print("[PASS] band_count OK")
        if any_range_fail:
            print(
                f"[WARN] value range suspicious for var={var} (p01/p99 out of [{hard_min},{hard_max}])."
            )

        # compact log
        print(f"[INFO] size={ds.RasterXSize}x{ds.RasterYSize} bands={bands}")
        print("[INFO] first band descriptions:", descs)
        for bi, st in sampled:
            print(
                f"  band {bi:03d}: fill={st['fill_ratio']:.6f}  "
                f"min={st['min']:.3f} max={st['max']:.3f} mean={st['mean']:.3f} "
                f"p01={st['p01']:.3f} p99={st['p99']:.3f}"
            )

        rows.append(
            {
                "path": p.as_posix(),
                "var": var,
                "xsize": int(ds.RasterXSize),
                "ysize": int(ds.RasterYSize),
                "bands": int(bands),
                "expect_bands": int(args.expect_bands),
                "band_count_ok": bool(bands == args.expect_bands),
                "grid_ok": grid_ok,
                "grid_msg": grid_msg,
                "first_descs": str(descs),
                "sample_bands": str(idxs),
                "range_hard_min": hard_min,
                "range_hard_max": hard_max,
                "range_warn": bool(any_range_fail),
                "b1_fill_ratio": sampled[0][1]["fill_ratio"] if sampled else math.nan,
                "b1_min": sampled[0][1]["min"] if sampled else math.nan,
                "b1_max": sampled[0][1]["max"] if sampled else math.nan,
                "b1_mean": sampled[0][1]["mean"] if sampled else math.nan,
                "b1_p01": sampled[0][1]["p01"] if sampled else math.nan,
                "b1_p99": sampled[0][1]["p99"] if sampled else math.nan,
            }
        )

    # write CSV
    if rows:
        import csv

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("=" * 88)
        print(f"[DONE] wrote {out_csv} rows={len(rows)}")
        print("=" * 88)
    else:
        print("[DONE] no rows (all skipped?)")


if __name__ == "__main__":
    main()
