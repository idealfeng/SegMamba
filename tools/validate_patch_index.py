from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()
# 索引patch检查

@dataclass(frozen=True)
class Grid:
    xsize: int
    ysize: int
    gt: tuple[float, float, float, float, float, float]
    wkt: str


def _open(path: Path) -> gdal.Dataset:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return ds


def _grid(ds: gdal.Dataset) -> Grid:
    return Grid(
        xsize=int(ds.RasterXSize),
        ysize=int(ds.RasterYSize),
        gt=tuple(float(x) for x in ds.GetGeoTransform()),
        wkt=(ds.GetProjectionRef() or "").strip(),
    )


def _grid_equal(a: Grid, b: Grid, tol_origin=1e-6, tol_px=1e-9) -> tuple[bool, str]:
    if a.xsize != b.xsize or a.ysize != b.ysize:
        return False, "size mismatch"
    if abs(a.gt[1] - b.gt[1]) > tol_px or abs(a.gt[5] - b.gt[5]) > tol_px:
        return False, "pixel size mismatch"
    if abs(a.gt[0] - b.gt[0]) > tol_origin or abs(a.gt[3] - b.gt[3]) > tol_origin:
        return False, "origin mismatch"
    if abs(b.gt[2]) > tol_px or abs(b.gt[4]) > tol_px:
        return False, "rotation not ~0"
    if a.wkt and b.wkt and a.wkt != b.wkt:
        return False, "CRS mismatch"
    return True, "PASS"


def _resolve_path(base_dir: Path, p: str) -> Path:
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return Path("")
    # CSV stores Windows-style relative paths like ..\data\...
    p = p.replace("/", "\\")
    path = Path(p)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate patch index CSV: file existence, window bounds, and ERA5/MODIS grid alignment."
    )
    ap.add_argument(
        "--csv",
        dest="csv_path",
        default="data/patch_index/filter_patches_no_overlap.csv",
        help="Patch index CSV path (default: data/patch_index/filter_patches_no_overlap.csv).",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="Base dir for resolving relative paths in CSV (default: <repo>/tools).",
    )
    ap.add_argument(
        "--out",
        default="tools/qc_out/patch_index_validate.txt",
        help="Write a human-readable report (default: tools/qc_out/patch_index_validate.txt).",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional limit for quick runs (0 = no limit).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    csv_path = _resolve_path(Path.cwd(), args.csv_path)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    base_dir = _resolve_path(Path.cwd(), args.base) if args.base else script_dir
    out_path = _resolve_path(Path.cwd(), args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    required = {
        "modis_path",
        "era5_temp_path",
        "era5_prcp_path",
        "group_id",
        "xoff",
        "yoff",
        "patch",
        "stride",
        "valid_ratio",
        "keep_reason",
    }

    n_rows = 0
    n_err = 0
    n_warn = 0

    missing_files: list[str] = []
    bad_windows: list[str] = []
    grid_mismatch: list[str] = []
    dup_windows: list[str] = []

    by_group_windows: dict[str, set[tuple[int, int]]] = {}
    modis_grid_cache: dict[Path, Grid] = {}
    era5_grid_cache: dict[Path, Grid] = {}
    modis_band_cache: dict[Path, int] = {}
    era5_band_cache: dict[Path, int] = {}

    valid_ratios: list[float] = []
    temp_nodata: list[float] = []
    prcp_nodata: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {csv_path}")
        missing_cols = sorted(required - set(reader.fieldnames))
        if missing_cols:
            raise RuntimeError(f"CSV missing columns: {missing_cols}")

        for row in reader:
            n_rows += 1
            if args.max_rows and n_rows > args.max_rows:
                break

            modis_path = _resolve_path(base_dir, row["modis_path"])
            era5_temp_path = _resolve_path(base_dir, row["era5_temp_path"])
            era5_prcp_path = _resolve_path(base_dir, row["era5_prcp_path"])
            group_id = row.get("group_id", "")

            try:
                xoff = int(float(row["xoff"]))
                yoff = int(float(row["yoff"]))
                patch = int(float(row["patch"]))
                stride = int(float(row["stride"]))
            except Exception:
                n_err += 1
                bad_windows.append(f"bad numeric fields: row={n_rows} group_id={group_id}")
                continue

            # duplicates within group
            s = by_group_windows.setdefault(group_id, set())
            if (xoff, yoff) in s:
                n_err += 1
                dup_windows.append(f"duplicate window: {group_id} xoff={xoff} yoff={yoff}")
            else:
                s.add((xoff, yoff))

            # existence
            for p in (modis_path, era5_temp_path, era5_prcp_path):
                if not p.exists():
                    n_err += 1
                    missing_files.append(str(p))

            if not (modis_path.exists() and era5_temp_path.exists() and era5_prcp_path.exists()):
                continue

            # open MODIS and check window bounds
            if modis_path not in modis_grid_cache:
                ds_m = _open(modis_path)
                modis_grid_cache[modis_path] = _grid(ds_m)
                modis_band_cache[modis_path] = int(ds_m.RasterCount)

            gm = modis_grid_cache[modis_path]
            if xoff < 0 or yoff < 0 or (xoff + patch) > gm.xsize or (yoff + patch) > gm.ysize:
                n_err += 1
                bad_windows.append(
                    f"out of bounds: {group_id} xoff={xoff} yoff={yoff} patch={patch} "
                    f"tile={gm.xsize}x{gm.ysize}"
                )

            if stride <= 0 or patch <= 0:
                n_err += 1
                bad_windows.append(f"invalid patch/stride: {group_id} patch={patch} stride={stride}")

            # collect stats (optional columns might be absent on old CSVs)
            try:
                valid_ratios.append(float(row.get("valid_ratio", "nan")))
            except Exception:
                pass
            try:
                if "temp_nodata_ratio_sample" in row and row["temp_nodata_ratio_sample"] != "":
                    temp_nodata.append(float(row["temp_nodata_ratio_sample"]))
                if "prcp_nodata_ratio_sample" in row and row["prcp_nodata_ratio_sample"] != "":
                    prcp_nodata.append(float(row["prcp_nodata_ratio_sample"]))
            except Exception:
                pass

            # grid alignment: ERA5 aligned must match MODIS grid exactly
            for era5_path in (era5_temp_path, era5_prcp_path):
                if era5_path not in era5_grid_cache:
                    ds_e = _open(era5_path)
                    era5_grid_cache[era5_path] = _grid(ds_e)
                    era5_band_cache[era5_path] = int(ds_e.RasterCount)
                ok, msg = _grid_equal(gm, era5_grid_cache[era5_path])
                if not ok:
                    n_err += 1
                    grid_mismatch.append(f"{era5_path} vs {modis_path}: {msg}")

            # band sanity
            if modis_band_cache.get(modis_path, 0) < 360:
                n_warn += 1
                # not fatal for some variants, but very suspicious for 6ch_fixed181
            if era5_band_cache.get(era5_temp_path, 0) != 181:
                n_warn += 1
            if era5_band_cache.get(era5_prcp_path, 0) != 181:
                n_warn += 1

    # summarize
    def _stat(xs: list[float]) -> tuple[float, float, float] | None:
        ys = [x for x in xs if x == x]  # drop NaN
        if not ys:
            return None
        ys.sort()
        return ys[0], ys[len(ys) // 2], ys[-1]

    vr = _stat(valid_ratios)
    tn = _stat(temp_nodata)
    pn = _stat(prcp_nodata)

    lines: list[str] = []
    lines.append(f"repo_root={repo_root}")
    lines.append(f"csv={csv_path}")
    lines.append(f"base={base_dir}")
    lines.append(f"rows_checked={n_rows}")
    lines.append(f"errors={n_err} warnings={n_warn}")
    if vr:
        lines.append(f"valid_ratio(min/median/max)={vr[0]:.6f}/{vr[1]:.6f}/{vr[2]:.6f}")
    if tn:
        lines.append(f"temp_nodata_ratio_sample(min/median/max)={tn[0]:.6f}/{tn[1]:.6f}/{tn[2]:.6f}")
    if pn:
        lines.append(f"prcp_nodata_ratio_sample(min/median/max)={pn[0]:.6f}/{pn[1]:.6f}/{pn[2]:.6f}")

    if missing_files:
        lines.append("")
        lines.append(f"[MISSING_FILES] n={len(missing_files)} (show first 20)")
        lines.extend(missing_files[:20])
    if bad_windows:
        lines.append("")
        lines.append(f"[BAD_WINDOWS] n={len(bad_windows)} (show first 20)")
        lines.extend(bad_windows[:20])
    if dup_windows:
        lines.append("")
        lines.append(f"[DUP_WINDOWS] n={len(dup_windows)} (show first 20)")
        lines.extend(dup_windows[:20])
    if grid_mismatch:
        lines.append("")
        lines.append(f"[GRID_MISMATCH] n={len(grid_mismatch)} (show first 20)")
        lines.extend(grid_mismatch[:20])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[WROTE] {out_path}")
    print(f"[SUMMARY] rows={n_rows} errors={n_err} warnings={n_warn}")

    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

