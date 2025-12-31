import os
import re
import csv
import glob
import numpy as np
from osgeo import gdal
"""
python filter_patches_no_overlap.py --root ..\data\raw_data\QinghaiTibet --out ..\data\patch_index\patch_index_core_no_overlap.csv
"""
# ===== Fixed layout you validated =====
T = 181
B_NDSI_1 = 1
B_MASK_1 = 1 + T
B_ELEV = 2 * T + 1  # band 363
# ======================================


def parse_group_and_offsets(path: str):
    """
    name like:
    Tibetan_SnowCover_2010_2011_6ch_fixed181_OFFICIAL-0000000000-0000002560.tif
    offsets are (yoff, xoff) in pixels
    """
    base = os.path.basename(path)
    group_id = base.split("-")[0]
    m = re.search(r"-(\d{10})-(\d{10})\.tif$", base)
    yoff = int(m.group(1)) if m else 0
    xoff = int(m.group(2)) if m else 0
    return group_id, yoff, xoff


def integral_image(a: np.ndarray) -> np.ndarray:
    # (H,W) -> (H+1,W+1)
    return (
        np.pad(a, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        .cumsum(0)
        .cumsum(1)
    )


def rect_sum(ii: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    y1 = y0 + h
    x1 = x0 + w
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def main(
    root,
    out_csv,
    patch=256,
    min_valid_ratio=0.10,
    max_elev_void_ratio=0.05,
    max_elev0_ratio=0.001,
):
    gdal.UseExceptions()

    tifs = sorted(glob.glob(os.path.join(root, "*.tif")))
    if not tifs:
        raise SystemExit(f"[ERROR] No tif found in: {root}")

    rows = []
    total_candidates = 0
    total_kept = 0

    print("============================================================")
    print("[FILTER] No-overlap patch filtering")
    print(f"  root={root}")
    print(f"  patch={patch}  stride={patch} (no overlap)")
    print(
        f"  rules: valid_ratio>={min_valid_ratio}, elev_void<={max_elev_void_ratio}, elev0<={max_elev0_ratio}"
    )
    print("============================================================")

    for tif in tifs:
        ds = gdal.Open(tif, gdal.GA_ReadOnly)
        if ds is None:
            raise SystemExit(f"[ERROR] Cannot open: {tif}")

        group_id, tile_yoff, tile_xoff = parse_group_and_offsets(tif)
        W, H = ds.RasterXSize, ds.RasterYSize

        # enumerate full windows only
        nx = (W - patch) // patch + 1 if W >= patch else 0
        ny = (H - patch) // patch + 1 if H >= patch else 0
        n_candidates_tile = nx * ny
        total_candidates += n_candidates_tile

        # 1) validDays per pixel = sum_t (MASK==1)
        validDays = np.zeros((H, W), dtype=np.uint16)
        for d in range(T):
            m = ds.GetRasterBand(B_MASK_1 + d).ReadAsArray()
            validDays += m == 1

        # 2) static elev
        elev = ds.GetRasterBand(B_ELEV).ReadAsArray().astype(np.int16)

        # 3) integral images for fast patch stats
        ii_v = integral_image(validDays.astype(np.uint32))
        ii_vd = integral_image((elev == -9999).astype(np.uint8))
        ii_v0 = integral_image((elev == 0).astype(np.uint8))

        kept_tile = 0

        for iy in range(ny):
            y0 = iy * patch
            for ix in range(nx):
                x0 = ix * patch
                area = patch * patch

                v_sum = rect_sum(
                    ii_v, x0, y0, patch, patch
                )  # sum(validDays) over pixels
                v_mean = v_sum / area
                valid_ratio = v_mean / T

                elev_void_ratio = rect_sum(ii_vd, x0, y0, patch, patch) / area
                elev0_ratio = rect_sum(ii_v0, x0, y0, patch, patch) / area

                # ===== Core rules =====
                if valid_ratio < min_valid_ratio:
                    continue
                if elev_void_ratio > max_elev_void_ratio:
                    continue
                if elev0_ratio > max_elev0_ratio:
                    continue
                # ======================

                # global coords (in pixel grid of the stitched group)
                gx0 = tile_xoff + x0
                gy0 = tile_yoff + y0

                patch_id = f"{group_id}-y{gy0:06d}-x{gx0:06d}"

                rows.append(
                    {
                        "patch_id": patch_id,
                        "group_id": group_id,
                        "tif_path": tif,
                        "tile_xoff": tile_xoff,
                        "tile_yoff": tile_yoff,
                        "x0": x0,
                        "y0": y0,
                        "global_x0": gx0,
                        "global_y0": gy0,
                        "validDays_mean": float(v_mean),
                        "valid_ratio": float(valid_ratio),
                        "r255_est": float(1.0 - valid_ratio),
                        "static_elev_ratio_eq_-9999": float(elev_void_ratio),
                        "static_elev_ratio_eq_0": float(elev0_ratio),
                    }
                )
                kept_tile += 1

        total_kept += kept_tile

        print(os.path.basename(tif))
        print(
            f"  size={W}x{H}  candidates={n_candidates_tile}  kept={kept_tile}  keep_rate={kept_tile/max(n_candidates_tile,1):.3f}"
        )
        ds = None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        # still write header to be explicit
        fieldnames = [
            "patch_id",
            "group_id",
            "tif_path",
            "tile_xoff",
            "tile_yoff",
            "x0",
            "y0",
            "global_x0",
            "global_y0",
            "validDays_mean",
            "valid_ratio",
            "r255_est",
            "static_elev_ratio_eq_-9999",
            "static_elev_ratio_eq_0",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    print("------------------------------------------------------------")
    print(
        f"[SUMMARY] candidates={total_candidates}  kept={total_kept}  keep_rate={total_kept/max(total_candidates,1):.3f}"
    )
    print(f"[OUTPUT ] {out_csv}")
    print("============================================================")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"..\data\raw_data\QinghaiTibet")
    ap.add_argument(
        "--out",
        required=True,
        help=r"..\data\patch_index\patch_index_core_no_overlap.csv",
    )
    ap.add_argument("--patch", type=int, default=256)

    ap.add_argument("--min_valid_ratio", type=float, default=0.10)
    ap.add_argument("--max_elev_void_ratio", type=float, default=0.05)
    ap.add_argument("--max_elev0_ratio", type=float, default=0.001)

    args = ap.parse_args()
    main(
        root=args.root,
        out_csv=args.out,
        patch=args.patch,
        min_valid_ratio=args.min_valid_ratio,
        max_elev_void_ratio=args.max_elev_void_ratio,
        max_elev0_ratio=args.max_elev0_ratio,
    )
