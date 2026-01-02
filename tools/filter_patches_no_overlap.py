import os
import re
import csv
import argparse
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

"""
python filter_patches_no_overlap.py `
--root_modis ..\data\raw_data\QinghaiTibet `
--root_era5  ..\data\raw_data\QinghaiTibet_ERA5L\aligned_to_modis_tiles `
--out        ..\data\patch_index\filter_patches_no_overlap.csv `
--min_valid_ratio 0.10
"""

def find_matching_era5(aligned_dir: str, modis_path: str, kind: str) -> str:
    """
    kind: 'temp' or 'prcp'
    We try multiple suffix patterns, prefer ALIGN2 > ALIGN.
    """
    base = os.path.basename(modis_path)
    base_noext = os.path.splitext(base)[0]

    if kind == "temp":
        candidates = [
            f"{base_noext}.ERA5T2mC.ALIGN2.vrt",
            f"{base_noext}.ERA5T2mC.ALIGN2.tif",
            f"{base_noext}.ERA5T2mC.ALIGN.vrt",
            f"{base_noext}.ERA5T2mC.ALIGN.tif",
        ]
    else:
        candidates = [
            f"{base_noext}.ERA5PrcpMM.ALIGN2.vrt",
            f"{base_noext}.ERA5PrcpMM.ALIGN2.tif",
            f"{base_noext}.ERA5PrcpMM.ALIGN.vrt",
            f"{base_noext}.ERA5PrcpMM.ALIGN.tif",
        ]

    for name in candidates:
        p = os.path.join(aligned_dir, name)
        if os.path.exists(p):
            return p

    # fallback: fuzzy match (in case group_id changed a bit)
    patt = re.escape(base_noext) + (
        r"\.ERA5T2mC\..*?\.(vrt|tif)$"
        if kind == "temp"
        else r"\.ERA5PrcpMM\..*?\.(vrt|tif)$"
    )
    for fn in os.listdir(aligned_dir):
        if re.match(patt, fn):
            return os.path.join(aligned_dir, fn)

    return ""


def read_window(ds: gdal.Dataset, band_list, xoff, yoff, w, h) -> np.ndarray:
    # GDAL band_list is 1-based
    arr = ds.ReadAsArray(xoff, yoff, w, h, band_list=band_list)
    if arr is None:
        raise RuntimeError("ReadAsArray returned None")
    return arr


def nodata_mask(arr: np.ndarray, nodata_value=None) -> np.ndarray:
    # ERA5 float data: nodata is often -9999; be robust:
    if nodata_value is not None:
        return arr == nodata_value
    return arr <= -9000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_modis",
        required=True,
        help="MODIS 6ch dir, e.g. ..\\data\\raw_data\\QinghaiTibet",
    )
    ap.add_argument(
        "--root_era5",
        required=True,
        help="ERA5 aligned dir, e.g. ..\\data\\raw_data\\QinghaiTibet_ERA5L\\aligned_to_modis_tiles",
    )
    ap.add_argument("--out", required=True, help="Output patch index CSV path")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)  # no overlap default
    ap.add_argument("--min_valid_ratio", type=float, default=0.10)
    ap.add_argument("--max_elev_void_ratio", type=float, default=0.05)
    ap.add_argument("--max_elev0_ratio", type=float, default=0.001)

    # optional strict ERA5 invasion threshold (set negative to disable)
    ap.add_argument("--max_era5_invasion", type=float, default=-1.0)

    # sample bands for ERA5 stats (1-based)
    ap.add_argument("--sample_bands", type=str, default="1,91,181")

    args = ap.parse_args()

    patch = args.patch
    stride = args.stride
    sample_bands = [int(x.strip()) for x in args.sample_bands.split(",") if x.strip()]

    # MODIS band layout (1-based):
    # 1..181 NDSI, 182..362 MASK, 363 Elev, 364 Slope_x100, 365 North_x10000, 366 NDVI_x10000
    MASK_BANDS = list(range(182, 363))
    ELEV_BAND = 363

    # For day k in [1..181], MASK band index = 181 + k
    def mask_band_for_day(k1: int) -> int:
        return 181 + k1

    # gather MODIS tifs
    modis_tifs = []
    for root, _, files in os.walk(args.root_modis):
        for fn in files:
            if fn.lower().endswith(".tif") and "OFFICIAL" in fn and ".ALIGN" not in fn:
                modis_tifs.append(os.path.join(root, fn))
    modis_tifs.sort()

    if not modis_tifs:
        raise RuntimeError(f"No MODIS OFFICIAL tif found under {args.root_modis}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = [
        "modis_path",
        "era5_temp_path",
        "era5_prcp_path",
        "group_id",
        "xoff",
        "yoff",
        "patch",
        "stride",
        "valid_ratio",
        "elev_void_ratio",
        "elev0_ratio",
        "temp_nodata_ratio_sample",
        "prcp_nodata_ratio_sample",
        "temp_invasion_ratio_sample",
        "prcp_invasion_ratio_sample",
        "keep_reason",
    ]

    total_candidates = 0
    total_kept = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for modis_path in modis_tifs:
            base_noext = os.path.splitext(os.path.basename(modis_path))[0]
            group_id = (
                base_noext  # keep simple; your existing scripts use filename-based ids
            )

            era5_temp = find_matching_era5(args.root_era5, modis_path, "temp")
            era5_prcp = find_matching_era5(args.root_era5, modis_path, "prcp")

            if not era5_temp or not era5_prcp:
                print(f"[SKIP] missing ERA5 match for: {modis_path}")
                continue

            ds_m = gdal.Open(modis_path, gdal.GA_ReadOnly)
            ds_t = gdal.Open(era5_temp, gdal.GA_ReadOnly)
            ds_p = gdal.Open(era5_prcp, gdal.GA_ReadOnly)

            w = ds_m.RasterXSize
            h = ds_m.RasterYSize

            # quick sanity on ERA5 band count
            if ds_t.RasterCount != 181 or ds_p.RasterCount != 181:
                print(
                    f"[WARN] ERA5 band count !=181: {era5_temp}({ds_t.RasterCount}), {era5_prcp}({ds_p.RasterCount})"
                )

            nx = (w - patch) // stride + 1 if w >= patch else 0
            ny = (h - patch) // stride + 1 if h >= patch else 0
            candidates = nx * ny
            kept = 0

            for iy in range(ny):
                yoff = iy * stride
                for ix in range(nx):
                    xoff = ix * stride
                    total_candidates += 1

                    # 1) valid_ratio from MASK (181 bands)
                    mask_cube = read_window(
                        ds_m, MASK_BANDS, xoff, yoff, patch, patch
                    )  # (181, H, W)
                    valid_ratio = float((mask_cube == 1).mean())

                    # 2) elevation ratios
                    elev = read_window(ds_m, [ELEV_BAND], xoff, yoff, patch, patch)
                    if elev.ndim == 3:
                        elev = elev[0]
                    elev_void_ratio = float((elev == -9999).mean())
                    elev0_ratio = float((elev == 0).mean())

                    # 3) ERA5 sample nodata + invasion (sample days only)
                    temp_nodata_ratio = 0.0
                    prcp_nodata_ratio = 0.0
                    temp_invasion_ratio = 0.0
                    prcp_invasion_ratio = 0.0

                    for k in sample_bands:
                        if k < 1 or k > 181:
                            continue
                        tb = read_window(ds_t, [k], xoff, yoff, patch, patch)
                        pb = read_window(ds_p, [k], xoff, yoff, patch, patch)
                        if tb.ndim == 3:
                            tb = tb[0]
                        if pb.ndim == 3:
                            pb = pb[0]

                        # era5 nodata
                        tmask = nodata_mask(tb)
                        pmask = nodata_mask(pb)

                        temp_nodata_ratio += float(tmask.mean())
                        prcp_nodata_ratio += float(pmask.mean())

                        # modis valid area for this day
                        mb = mask_band_for_day(k)
                        mday = read_window(ds_m, [mb], xoff, yoff, patch, patch)
                        if mday.ndim == 3:
                            mday = mday[0]
                        valid_area = mday == 1
                        denom = int(valid_area.sum())
                        if denom > 0:
                            temp_invasion_ratio += float(
                                (tmask & valid_area).sum() / denom
                            )
                            prcp_invasion_ratio += float(
                                (pmask & valid_area).sum() / denom
                            )
                        else:
                            # if no valid pixels that day, invasion not meaningful; treat as 0
                            temp_invasion_ratio += 0.0
                            prcp_invasion_ratio += 0.0

                    denom_days = max(1, len([k for k in sample_bands if 1 <= k <= 181]))
                    temp_nodata_ratio /= denom_days
                    prcp_nodata_ratio /= denom_days
                    temp_invasion_ratio /= denom_days
                    prcp_invasion_ratio /= denom_days

                    # 4) filtering rules
                    keep = True
                    reason = "PASS"

                    if valid_ratio < args.min_valid_ratio:
                        keep = False
                        reason = f"FAIL_valid_ratio<{args.min_valid_ratio}"
                    elif elev_void_ratio > args.max_elev_void_ratio:
                        keep = False
                        reason = f"FAIL_elev_void>{args.max_elev_void_ratio}"
                    elif elev0_ratio > args.max_elev0_ratio:
                        keep = False
                        reason = f"FAIL_elev0>{args.max_elev0_ratio}"

                    if keep and args.max_era5_invasion >= 0.0:
                        max_inv = max(temp_invasion_ratio, prcp_invasion_ratio)
                        if max_inv > args.max_era5_invasion:
                            keep = False
                            reason = f"FAIL_era5_invasion>{args.max_era5_invasion}"

                    if keep:
                        kept += 1
                        total_kept += 1

                        wr.writerow(
                            [
                                modis_path,
                                era5_temp,
                                era5_prcp,
                                group_id,
                                xoff,
                                yoff,
                                patch,
                                stride,
                                f"{valid_ratio:.6f}",
                                f"{elev_void_ratio:.6f}",
                                f"{elev0_ratio:.6f}",
                                f"{temp_nodata_ratio:.6f}",
                                f"{prcp_nodata_ratio:.6f}",
                                f"{temp_invasion_ratio:.6f}",
                                f"{prcp_invasion_ratio:.6f}",
                                reason,
                            ]
                        )

            print(f"{os.path.basename(modis_path)}")
            print(
                f"  size={w}x{h}  candidates={candidates}  kept={kept}  keep_rate={(kept/max(1,candidates)):.3f}"
            )

            ds_m = None
            ds_t = None
            ds_p = None

    print("------------------------------------------------------------")
    print(
        f"[SUMMARY] candidates={total_candidates}  kept={total_kept}  keep_rate={(total_kept/max(1,total_candidates)):.3f}"
    )
    print(f"[OUTPUT ] {args.out}")


if __name__ == "__main__":
    main()
