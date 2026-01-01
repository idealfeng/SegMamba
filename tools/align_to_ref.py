import argparse, os
from osgeo import gdal

gdal.UseExceptions()


def ref_params(ref_path: str):
    ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(ref_path)

    gt = ds.GetGeoTransform()
    x0, px, _, y0, _, py = gt
    w, h = ds.RasterXSize, ds.RasterYSize
    proj = ds.GetProjection()

    xmin = x0
    xmax = x0 + px * w
    ymax = y0
    ymin = y0 + py * h  # py negative
    ds = None
    return (xmin, ymin, xmax, ymax), (w, h), proj


def warp_to_ref(src_path: str, ref_path: str, out_path: str):
    bounds, (w, h), proj = ref_params(ref_path)

    opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=proj,
        outputBounds=bounds,
        width=w,
        height=h,
        # 关键：不要 targetAlignedPixels（不要吸附格网）
        targetAlignedPixels=False,
        resampleAlg="near",
        multithread=True,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdal.Warp(out_path, src_path, options=opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warp_to_ref(args.src, args.ref, args.out)
    print("[OK] aligned v2:", args.out)


if __name__ == "__main__":
    main()
