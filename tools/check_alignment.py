import argparse
import csv
import os
import re
from pathlib import Path

from osgeo import gdal
# 检查原始数据是否对齐

def parse_offsets_from_name(path: str):
    # ...-0000000000-0000002560.tif  -> yoff=0, xoff=2560 (pixel offsets)
    base = os.path.basename(path)
    m = re.search(r"-(\d{10})-(\d{10})\.tif$", base)
    if not m:
        return None, None
    yoff = int(m.group(1))
    xoff = int(m.group(2))
    return yoff, xoff


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


def main(root: str | None, out_csv: str | None) -> int:
    gdal.UseExceptions()

    root_path = resolve_root(root)
    tifs = iter_tifs(root_path)
    if not tifs:
        raise SystemExit(f"No tif found under: {root_path}")

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "qc_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = Path(out_csv) if out_csv else (out_dir / "alignment_tiles.csv")
    if not out_csv_path.is_absolute():
        out_csv_path = (Path.cwd() / out_csv_path).resolve()

    print(f"[ALIGN] root={root_path}  tifs={len(tifs)}")
    print(f"[ALIGN] output={out_csv_path}")

    infos = []
    for p in tifs:
        ds = gdal.Open(str(p), gdal.GA_ReadOnly)
        if ds is None:
            raise SystemExit(f"Cannot open: {p}")
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()
        xsize, ysize = int(ds.RasterXSize), int(ds.RasterYSize)
        yoff, xoff = parse_offsets_from_name(str(p))
        infos.append((p, gt, proj, xsize, ysize, yoff, xoff))

    # Choose reference tile: prefer offset (0,0), else the smallest (yoff,xoff), else first.
    def _ref_key(it):
        p, _gt, _proj, _xsize, _ysize, yoff, xoff = it
        if yoff is None or xoff is None:
            return (1, 0, 0, str(p))
        return (0, int(yoff), int(xoff), str(p))

    ref = min(infos, key=_ref_key)
    ref_p, ref_gt, ref_proj, _ref_w, _ref_h, ref_yoff, ref_xoff = ref
    ref_px = (ref_gt[1], ref_gt[5], ref_gt[2], ref_gt[4])

    origin_x0, origin_y0 = ref_gt[0], ref_gt[3]
    px_w, px_h = ref_gt[1], ref_gt[5]
    if ref_yoff is None or ref_xoff is None:
        # If reference filename doesn't encode offsets, adjacency checks still work for other tiles
        # by using their own offsets relative to this reference origin.
        ref_yoff, ref_xoff = 0, 0

    rows = []
    ok_all = True
    for p, gt, proj, xsize, ysize, yoff, xoff in infos:
        crs_match = proj == ref_proj
        px = (gt[1], gt[5], gt[2], gt[4])
        px_match = all(abs(px[i] - ref_px[i]) <= 1e-9 for i in range(4))

        origin_match = None
        warn_origin = False
        if yoff is not None and xoff is not None:
            exp_x = origin_x0 + (xoff - ref_xoff) * px_w
            exp_y = origin_y0 + (yoff - ref_yoff) * px_h
            tol_x = abs(px_w) * 1e-6
            tol_y = abs(px_h) * 1e-6
            origin_match = abs(gt[0] - exp_x) <= tol_x and abs(gt[3] - exp_y) <= tol_y
            warn_origin = not origin_match

        passed = crs_match and px_match and (origin_match is not False)
        ok_all = ok_all and passed

        rows.append(
            {
                "tif_path": p.as_posix(),
                "xsize": xsize,
                "ysize": ysize,
                "yoff": yoff,
                "xoff": xoff,
                "origin_x": gt[0],
                "origin_y": gt[3],
                "px_w": gt[1],
                "px_h": gt[5],
                "rot_x": gt[2],
                "rot_y": gt[4],
                "crs_match_ref": crs_match,
                "px_match_ref": px_match,
                "origin_match_ref": origin_match,
                "status": "WARN" if warn_origin else ("PASS" if passed else "FAIL"),
                "ref_tif": ref_p.as_posix(),
            }
        )

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("[ALIGN] PASS" if ok_all else "[ALIGN] FAIL", f"(wrote {out_csv_path})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate GeoTIFF tile grid alignment under raw_data/Tibet (or given root), and save a CSV report."
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Folder containing .tif files (default: auto-detect data/raw_data/Tibet).",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: tools/qc_out/alignment_tiles.csv).",
    )
    args = ap.parse_args()
    raise SystemExit(main(args.root, args.out_csv))
