# align_era5_to_modis_tiles.py
# Create aligned ERA5 rasters (VRT/GeoTIFF) for each MODIS tile (reference grid),
# and save a CSV report for the whole batch.
#
# Default input folders:
# - MODIS tiles: data/raw_data/QinghaiTibet (fallbacks: data/raw_data/Tibet, data/raw_data)
# - ERA5-Land: data/raw_data/QinghaiTibet_ERA5L

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

WINTER_RE = re.compile(r"(\d{4})_(\d{4})")
ERA5_PREFIX_RE = re.compile(r"^(?P<prefix>.+?)_ERA5L_", re.IGNORECASE)


def _open(path: str):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return ds


def _ref_bounds_gt(ds):
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    # gt = (originX, pxW, rotX, originY, rotY, pxH) with pxH negative
    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * xsize
    miny = gt[3] + gt[5] * ysize
    return (minx, miny, maxx, maxy), gt


def _grid_equal(ref_ds, ds, tol_origin=1e-6, tol_px=1e-9):
    ref_gt = ref_ds.GetGeoTransform()
    gt = ds.GetGeoTransform()

    if ref_ds.RasterXSize != ds.RasterXSize or ref_ds.RasterYSize != ds.RasterYSize:
        return False, "size mismatch"

    if abs(ref_gt[1] - gt[1]) > tol_px or abs(ref_gt[5] - gt[5]) > tol_px:
        return False, "pixel size mismatch"

    if abs(ref_gt[0] - gt[0]) > tol_origin or abs(ref_gt[3] - gt[3]) > tol_origin:
        return False, "origin mismatch"

    if abs(gt[2]) > tol_px or abs(gt[4]) > tol_px:
        return False, "rotation not ~0"

    ref_wkt = (ref_ds.GetProjectionRef() or "").strip()
    wkt = (ds.GetProjectionRef() or "").strip()
    if ref_wkt and wkt and (ref_wkt != wkt):
        return False, "CRS mismatch"

    return True, "PASS"


def _find_era5_files(era5_dir: Path, region_prefix: str, y0: str, y1: str):
    # expects names like:
    #   Tibetan_ERA5L_T2mC_native_2010_2011_181d.tif
    #   Tibetan_ERA5L_PrcpMM_native_2010_2011_181d.tif
    temp = era5_dir / f"{region_prefix}_ERA5L_T2mC_native_{y0}_{y1}_181d.tif"
    prcp = era5_dir / f"{region_prefix}_ERA5L_PrcpMM_native_{y0}_{y1}_181d.tif"
    return temp, prcp


def warp_to_ref(
    src_path: Path,
    ref_path: Path,
    out_path: Path,
    fmt: str,
    resample: str,
    nodata: float,
):
    ref_ds = _open(str(ref_path))
    bounds, _ = _ref_bounds_gt(ref_ds)
    ref_wkt = ref_ds.GetProjectionRef()
    width, height = ref_ds.RasterXSize, ref_ds.RasterYSize

    options = gdal.WarpOptions(
        format=fmt,
        dstSRS=ref_wkt,
        outputBounds=bounds,  # exact bounds from ref
        width=width,
        height=height,  # exact size from ref (prevents +1 pixel drift)
        resampleAlg=resample,
        srcNodata=nodata,
        dstNodata=nodata,
        multithread=True,
        creationOptions=(
            ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_SAFER"]
            if fmt.upper() in ("GTIFF", "COG")
            else []
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = gdal.Warp(
        destNameOrDestDS=str(out_path), srcDSOrSrcDSTab=str(src_path), options=options
    )
    if out is None:
        raise RuntimeError(f"gdal.Warp failed: {src_path} -> {out_path}")
    out.FlushCache()
    out = None

    # verify grid
    out_ds = _open(str(out_path))
    ok, msg = _grid_equal(ref_ds, out_ds)
    return ok, msg


def _resolve_root_from_repo(repo_root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    # user-provided relative paths should be relative to current working directory,
    # so the script works whether called from repo root or from tools/.
    return (Path.cwd() / p).resolve()


def _infer_region_prefix(era5_dir: Path) -> str:
    files = sorted(era5_dir.glob("*.tif"))
    for f in files:
        m = ERA5_PREFIX_RE.match(f.name)
        if m:
            return m.group("prefix")
    raise RuntimeError(f"Cannot infer region_prefix from tif names under: {era5_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modis_root",
        default=None,
        help="Folder containing MODIS tile tifs (reference grid). Default: data/raw_data/QinghaiTibet.",
    )
    ap.add_argument(
        "--era5_dir",
        default=None,
        help="Folder containing ERA5 exports (native GeoTIFF). Default: data/raw_data/QinghaiTibet_ERA5L.",
    )
    ap.add_argument(
        "--region_prefix",
        default=None,
        help="Prefix used in ERA5 filenames, e.g., Tibetan. If omitted, inferred from files in --era5_dir.",
    )
    ap.add_argument(
        "--out_root",
        default=None,
        help="Output folder for aligned VRT/GeoTIFF. Default: <era5_dir>/aligned_to_modis_tiles.",
    )
    ap.add_argument(
        "--format",
        default="VRT",
        choices=["VRT", "GTiff"],
        help="Default VRT (recommended).",
    )
    ap.add_argument(
        "--resample_temp",
        default="bilinear",
        help="Resampling for temp (bilinear recommended).",
    )
    ap.add_argument(
        "--resample_prcp",
        default="bilinear",
        help="Resampling for prcp (bilinear/average).",
    )
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument(
        "--glob", default="*.tif", help="MODIS tile glob pattern (default *.tif)."
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Optional substring filter for MODIS tile filenames (useful for a single year).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of MODIS tiles to process (0 = no limit).",
    )
    ap.add_argument(
        "--out-csv",
        dest="out_csv",
        default=None,
        help="Write a per-tile alignment report CSV (default: tools/qc_out/era5_align_to_modis.csv).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.modis_root is None:
        candidates = [
            repo_root / "data" / "raw_data" / "Tibet",
            repo_root / "data" / "raw_data" / "QinghaiTibet",
            repo_root / "data" / "raw_data",
        ]
        modis_root = next((c for c in candidates if c.exists()), candidates[0])
    else:
        modis_root = _resolve_root_from_repo(repo_root, args.modis_root)

    if args.era5_dir is None:
        era5_dir = repo_root / "data" / "raw_data" / "QinghaiTibet_ERA5L"
    else:
        era5_dir = _resolve_root_from_repo(repo_root, args.era5_dir)

    if not modis_root.exists():
        raise RuntimeError(f"modis_root not found: {modis_root}")
    if not era5_dir.exists():
        raise RuntimeError(f"era5_dir not found: {era5_dir}")

    region_prefix = args.region_prefix or _infer_region_prefix(era5_dir)
    out_root = (
        _resolve_root_from_repo(repo_root, args.out_root)
        if args.out_root
        else (era5_dir / "aligned_to_modis_tiles")
    )

    out_csv = (
        _resolve_root_from_repo(repo_root, args.out_csv)
        if args.out_csv
        else (script_dir / "qc_out" / "era5_align_to_modis.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    modis_tiles = sorted(modis_root.glob(args.glob))
    if not modis_tiles:
        raise RuntimeError(
            f"No MODIS tiles found under {modis_root} with glob={args.glob}"
        )

    print("=" * 88)
    print("[ALIGN ERA5 -> MODIS GRID]")
    print(f"  modis_root={modis_root}")
    print(f"  era5_dir={era5_dir}")
    print(f"  region_prefix={region_prefix}")
    print(f"  out_root={out_root}")
    print(f"  format={fmt}  nodata={args.nodata}")
    print("=" * 88)

    n_ok = 0
    n_fail = 0
    n_skip = 0
    rows: list[dict[str, str]] = []

    for ref_path in modis_tiles:
        name = ref_path.name
        if args.only and (args.only not in name):
            continue

        m = WINTER_RE.search(name)
        if not m:
            n_skip += 1
            continue

        y0, y1 = m.group(1), m.group(2)

        temp_src, prcp_src = _find_era5_files(era5_dir, region_prefix, y0, y1)
        if (not temp_src.exists()) or (not prcp_src.exists()):
            print(
                f"[SKIP] {name}  (missing ERA5 src: temp={temp_src.exists()} prcp={prcp_src.exists()})"
            )
            n_skip += 1
            continue

        stem = ref_path.stem
        ext = ".vrt" if fmt.upper() == "VRT" else ".tif"
        out_temp = out_root / f"{stem}.ERA5T2mC.ALIGN{ext}"
        out_prcp = out_root / f"{stem}.ERA5PrcpMM.ALIGN{ext}"

        try:
            ok1, msg1 = warp_to_ref(
                temp_src, ref_path, out_temp, fmt, args.resample_temp, args.nodata
            )
            ok2, msg2 = warp_to_ref(
                prcp_src, ref_path, out_prcp, fmt, args.resample_prcp, args.nodata
            )

            status = "PASS" if (ok1 and ok2) else "FAIL"
            print(f"[{status}] {name}")
            if not ok1:
                print(f"   temp: {msg1} -> {out_temp}")
            if not ok2:
                print(f"   prcp: {msg2} -> {out_prcp}")

            if ok1 and ok2:
                n_ok += 1
            else:
                n_fail += 1

            rows.append(
                {
                    "modis_ref": ref_path.as_posix(),
                    "era5_temp_src": temp_src.as_posix(),
                    "era5_prcp_src": prcp_src.as_posix(),
                    "out_temp": out_temp.as_posix(),
                    "out_prcp": out_prcp.as_posix(),
                    "status": status,
                    "temp_msg": msg1,
                    "prcp_msg": msg2,
                    "year0": y0,
                    "year1": y1,
                    "format": fmt,
                }
            )

        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {name}  error={e}")
            rows.append(
                {
                    "modis_ref": ref_path.as_posix(),
                    "era5_temp_src": temp_src.as_posix(),
                    "era5_prcp_src": prcp_src.as_posix(),
                    "out_temp": out_temp.as_posix(),
                    "out_prcp": out_prcp.as_posix(),
                    "status": "ERROR",
                    "temp_msg": str(e),
                    "prcp_msg": str(e),
                    "year0": y0,
                    "year1": y1,
                    "format": fmt,
                }
            )

        if args.limit and (n_ok + n_fail) >= args.limit:
            break

    print("-" * 88)
    print(
        f"[SUMMARY] ok={n_ok}  fail={n_fail}  skip={n_skip}  total_ref_tiles={len(modis_tiles)}"
    )

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[WROTE] {out_csv}")

    print("=" * 88)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
