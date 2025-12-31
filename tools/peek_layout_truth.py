import os
import re
import numpy as np
import rasterio
from datetime import date, timedelta

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


def mismatch_stats(ndsi: np.ndarray, mask: np.ndarray):
    mismatch = float(np.mean((mask == 0) != (ndsi == NDSI_FILL)))
    pseudo0 = float(np.mean((mask == 1) & (ndsi == NDSI_FILL)))
    mask0 = float(np.mean(mask == 0))
    ndsi255 = float(np.mean(ndsi == NDSI_FILL))
    return mismatch, pseudo0, mask0, ndsi255


def peek_one(path: str, day_index: int = 0):
    gid = group_id_from_path(path)
    yrs = years_from_gid(gid)
    day = winter_dates(yrs[0])[day_index] if yrs else f"idx{day_index}"
    print("=" * 96)
    print("tif:", path)
    print("group:", gid, "day:", day, "day_index:", day_index)

    with rasterio.open(path) as ds:
        print(
            "bands:",
            ds.count,
            "size:",
            ds.width,
            ds.height,
            "dtype band1:",
            ds.dtypes[0],
        )

        # 先把 band1/band2 的“长相”打印出来
        b1 = ds.read(1)
        b2 = ds.read(2)
        print(
            "band1 min/max:",
            int(b1.min()),
            int(b1.max()),
            "unique_head:",
            np.unique(b1)[:20],
        )
        print(
            "band2 min/max:",
            int(b2.min()),
            int(b2.max()),
            "unique_head:",
            np.unique(b2)[:20],
        )

        # 2-band 可能布局（6ch=366 => 181*2 + 4）
        D = 181
        total = ds.count
        static_count = 4
        dyn = total - static_count
        if dyn != 2 * D:
            print("[WARN] dyn bands != 362; this file may not be 6ch format.")
        # 布局定义：给 day_index 返回 (ndsi_band, mask_band)
        layouts = {
            "2x_interleaved_NDSI_MASK": lambda t: (2 * t + 1, 2 * t + 2),
            "2x_interleaved_MASK_NDSI": lambda t: (2 * t + 2, 2 * t + 1),
            "2x_grouped_NDSI_then_MASK": lambda t: (t + 1, D + t + 1),
            "2x_grouped_MASK_then_NDSI": lambda t: (D + t + 1, t + 1),
        }

        results = []
        for name, fn in layouts.items():
            ndsi_i, mask_i = fn(day_index)
            if ndsi_i < 1 or mask_i < 1 or ndsi_i > ds.count or mask_i > ds.count:
                continue
            ndsi = ds.read(ndsi_i)
            mask = ds.read(mask_i)

            # 让你一眼看出“谁像 NDSI、谁像 mask”
            u_mask = np.unique(mask)
            mask_like = u_mask.size <= 3 and set(u_mask.tolist()).issubset({0, 1})
            ndsi_like = (
                ndsi.min() >= 0 and ndsi.max() <= 255 and (np.mean(ndsi == 255) > 1e-4)
            )

            mismatch, pseudo0, mask0, ndsi255 = mismatch_stats(ndsi, mask)
            results.append(
                (
                    mismatch,
                    pseudo0,
                    name,
                    ndsi_i,
                    mask_i,
                    mask_like,
                    ndsi_like,
                    mask0,
                    ndsi255,
                )
            )

        results.sort(key=lambda x: (x[0], x[1]))
        print("-" * 96)
        print("Try layouts (sorted by mismatch):")
        for (
            mismatch,
            pseudo0,
            name,
            ndsi_i,
            mask_i,
            mask_like,
            ndsi_like,
            mask0,
            ndsi255,
        ) in results:
            print(
                f"{name:28s}  ndsi={ndsi_i:3d} mask={mask_i:3d}  "
                f"mismatch={mismatch:.9f} pseudo0={pseudo0:.9f}  "
                f"mask0={mask0:.6f} ndsi255={ndsi255:.6f}  "
                f"mask_like={mask_like} ndsi_like={ndsi_like}"
            )

        best = results[0] if results else None
        if best:
            print("-" * 96)
            print("BEST:", best[2], "mismatch=", best[0], "pseudo0=", best[1])
            if best[0] == 0.0 and best[1] == 0.0:
                print(
                    "=> PASS: mask and ndsi fill are perfectly aligned (as expected)."
                )
            else:
                print(
                    "=> FAIL: no layout achieves mismatch=0, which means the exported MASK is NOT derived from NDSI==255 (or file is not the intended export)."
                )


def main():
    root = r"data/raw_data/QinghaiTibet"
    # 扫该目录所有 tif
    tifs = []
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith(".tif"):
                tifs.append(os.path.join(dp, fn))
    tifs.sort()

    print("[FOUND]", len(tifs), "tifs")
    for p in tifs:
        peek_one(p, day_index=0)  # 20101101
        peek_one(p, day_index=90)  # 中间一天
        peek_one(p, day_index=180)  # 20110430


if __name__ == "__main__":
    main()
