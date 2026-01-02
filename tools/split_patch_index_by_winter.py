from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WINTER_RE = re.compile(r"(\d{4})_(\d{4})")


def _parse_list(s: str) -> list[str]:
    items = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split a patch_index CSV into train/val/test by winter (YYYY_YYYY in group_id)."
    )
    ap.add_argument(
        "--in",
        dest="in_csv",
        default="data/patch_index/filter_patches_no_overlap.csv",
        help="Input patch index CSV (default: data/patch_index/filter_patches_no_overlap.csv).",
    )
    ap.add_argument(
        "--out-dir",
        default="data/patch_index",
        help="Output directory (default: data/patch_index).",
    )
    ap.add_argument(
        "--train-winters",
        default="2010_2011,2011_2012,2012_2013,2013_2014,2014_2015,2015_2016,2016_2017,2017_2018,2018_2019,2019_2020,2020_2021",
        help="Comma-separated winters for train.",
    )
    ap.add_argument(
        "--val-winters",
        default="2021_2022,2022_2023",
        help="Comma-separated winters for val.",
    )
    ap.add_argument(
        "--test-winters",
        default="2023_2024,2024_2025",
        help="Comma-separated winters for test.",
    )
    ap.add_argument(
        "--prefix",
        default="scd_no_overlap",
        help="Output filename prefix (default: scd_no_overlap).",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = set(_parse_list(args.train_winters))
    val_set = set(_parse_list(args.val_winters))
    test_set = set(_parse_list(args.test_winters))

    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        raise RuntimeError("Overlap found between train/val/test winter sets.")

    with in_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {in_csv}")
        fieldnames = list(reader.fieldnames)

        rows_train = []
        rows_val = []
        rows_test = []
        rows_unknown = 0
        rows_drop = 0

        for row in reader:
            gid = row.get("group_id", "")
            m = WINTER_RE.search(gid)
            if not m:
                rows_unknown += 1
                continue
            winter = f"{m.group(1)}_{m.group(2)}"
            if winter in train_set:
                rows_train.append(row)
            elif winter in val_set:
                rows_val.append(row)
            elif winter in test_set:
                rows_test.append(row)
            else:
                rows_drop += 1

    def _write(name: str, rows: list[dict]):
        p = out_dir / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return p

    p_tr = _write(f"{args.prefix}_train.csv", rows_train)
    p_va = _write(f"{args.prefix}_val.csv", rows_val)
    p_te = _write(f"{args.prefix}_test.csv", rows_test)

    print(f"[WROTE] {p_tr} rows={len(rows_train)}")
    print(f"[WROTE] {p_va} rows={len(rows_val)}")
    print(f"[WROTE] {p_te} rows={len(rows_test)}")
    print(f"[INFO] unknown_winter_rows={rows_unknown} dropped_other_winters={rows_drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

