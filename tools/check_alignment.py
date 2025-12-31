import os
import glob
import re
from osgeo import gdal

"""
python check_alignment.py --root ..\data\raw_data\QinghaiTibet

"""
def parse_offsets_from_name(path: str):
    # ...-0000000000-0000002560.tif  -> yoff=0, xoff=2560 (pixel offsets)
    base = os.path.basename(path)
    m = re.search(r"-(\d{10})-(\d{10})\.tif$", base)
    if not m:
        return None, None
    yoff = int(m.group(1))
    xoff = int(m.group(2))
    return yoff, xoff


def main(root):
    tifs = sorted(glob.glob(os.path.join(root, "*.tif")))
    if not tifs:
        raise SystemExit(f"No tif found in: {root}")

    print(f"[ALIGN] Found {len(tifs)} tif(s) in {root}\n")

    infos = []
    for p in tifs:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        if ds is None:
            raise SystemExit(f"Cannot open: {p}")
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()
        xsize, ysize = ds.RasterXSize, ds.RasterYSize
        yoff, xoff = parse_offsets_from_name(p)

        infos.append((p, gt, proj, xsize, ysize, yoff, xoff))

        print(os.path.basename(p))
        print(f"  size={xsize}x{ysize}  offsets(y,x)={yoff},{xoff}")
        print(
            f"  origin=({gt[0]:.3f},{gt[3]:.3f})  px=({gt[1]:.6f},{gt[5]:.6f}) rot=({gt[2]:.6f},{gt[4]:.6f})"
        )
        print("")

    # Check identical CRS and pixel size / rotation
    ref_gt, ref_proj = infos[0][1], infos[0][2]
    ref_px = (ref_gt[1], ref_gt[5], ref_gt[2], ref_gt[4])

    ok = True
    for p, gt, proj, xsize, ysize, yoff, xoff in infos[1:]:
        if proj != ref_proj:
            print(f"[FAIL] CRS mismatch: {os.path.basename(p)}")
            ok = False
        px = (gt[1], gt[5], gt[2], gt[4])
        if any(abs(px[i] - ref_px[i]) > 1e-9 for i in range(4)):
            print(
                f"[FAIL] PixelSize/rotation mismatch: {os.path.basename(p)}  {px} != {ref_px}"
            )
            ok = False

    # Adjacency sanity check using filename offsets (most reliable in your pipeline)
    # If offsets exist, verify origin shift matches xoff/yoff * pixel size.
    gt0 = infos[0][1]
    origin_x0, origin_y0 = gt0[0], gt0[3]
    px_w, px_h = gt0[1], gt0[5]

    for p, gt, proj, xsize, ysize, yoff, xoff in infos:
        if yoff is None or xoff is None:
            continue
        exp_x = origin_x0 + xoff * px_w
        exp_y = origin_y0 + yoff * px_h
        if (
            abs(gt[0] - exp_x) > abs(px_w) * 1e-6
            or abs(gt[3] - exp_y) > abs(px_h) * 1e-6
        ):
            print(
                f"[WARN] Origin not consistent with filename offsets: {os.path.basename(p)}"
            )
            print(f"       gt_origin=({gt[0]}, {gt[3]}) expected=({exp_x}, {exp_y})")

    if ok:
        print("[PASS] CRS + pixel grid parameters consistent across tiles.")
    else:
        print(
            "[FAIL] Alignment checks failed. Fix export CRS/transform in GEE before patching."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="e.g. ..\\data\\raw_data\\QinghaiTibet"
    )
    args = ap.parse_args()
    main(args.root)
