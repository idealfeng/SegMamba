"""
LEVIR-CD 变化检测数据集加载器
"""
import os
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from config import cfg
from osgeo import gdal

gdal.UseExceptions()

EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def _find_existing(folder: Path, stem: str) -> Path:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing file for stem={stem} under {folder}")


def _read_rgb_any(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    # 16-bit tif -> 8-bit
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)

    # 保证3通道
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def _read_mask_any(path: Path) -> np.ndarray:
    m = np.array(Image.open(path))
    # 可能是 0/255 或 0/1 都统一成 0/1
    if m.dtype == np.uint16:
        m = (m >> 8).astype(np.uint8)
    m = (m > 0).astype(np.uint8)
    return m


class LEVIRCDDataset(Dataset):
    """LEVIR-CD变化检测数据集"""

    @staticmethod
    def _find_split_dir(root_dir: Path, split: str, max_depth: int = 3) -> Optional[Path]:
        """
        Find a directory that contains the given split folder.
        Handles wrappers like root/JL1-CD/JL1-CD/.../{train,test,...}.
        Returns the split directory path (i.e., .../<split>) or None.
        """
        root_dir = Path(root_dir)
        split = str(split)
        if (root_dir / split).is_dir():
            return root_dir / split

        queue: List[Tuple[Path, int]] = [(root_dir, 0)]
        hits: List[Tuple[int, Path]] = []
        while queue:
            cur, depth = queue.pop(0)
            candidate = cur / split
            if candidate.is_dir():
                hits.append((depth, candidate))
                continue
            if depth >= max_depth:
                continue
            try:
                children = [p for p in cur.iterdir() if p.is_dir() and not p.name.startswith(".")]
            except Exception:
                children = []
            for ch in children:
                queue.append((ch, depth + 1))

        if not hits:
            return None
        hits.sort(key=lambda x: x[0])
        return hits[0][1]

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        crop_size: int = 256
    ):
        """
        Args:
            root_dir: 数据集根目录 (包含train/val/test子目录)
            split: 数据集划分 ('train', 'val', 'test')
            transform: albumentations变换
            crop_size: 裁剪尺寸
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        # 设置路径
        split_dir = self.root_dir / split
        if not split_dir.is_dir():
            found = self._find_split_dir(self.root_dir, split)
            if found is not None:
                split_dir = found
        # Some datasets are packaged as root/split/split/{A,B,label} (e.g., SYSU-CD/train/train/A).
        if not (split_dir / "A").exists():
            nested = split_dir / split
            if (nested / "A").exists():
                split_dir = nested

        def _pick_dir(base: Path, candidates: List[str]) -> Optional[Path]:
            for name in candidates:
                p = base / name
                if p.is_dir():
                    return p
            return None

        # Different CD datasets use different folder names.
        # Prefer canonical A/B/label, but support common aliases (e.g., S2Looking uses Image1/Image2).
        img_a_dir = _pick_dir(split_dir, ["A", "Image1", "T1", "t1", "img1", "im1"])
        img_b_dir = _pick_dir(split_dir, ["B", "Image2", "T2", "t2", "img2", "im2"])
        label_dir = _pick_dir(split_dir, ["label", "Label", "GT", "gt", "mask", "masks", "labels", "label1", "label2"])

        # Extra fallback: root/split/split/{Image1,Image2,...}
        if img_a_dir is None or img_b_dir is None:
            nested = split_dir / split
            if nested.is_dir():
                img_a_dir = img_a_dir or _pick_dir(nested, ["A", "Image1", "T1", "t1", "img1", "im1"])
                img_b_dir = img_b_dir or _pick_dir(nested, ["B", "Image2", "T2", "t2", "img2", "im2"])
                label_dir = label_dir or _pick_dir(nested, ["label", "Label", "GT", "gt", "mask", "masks", "labels", "label1", "label2"])

        self.img_a_dir = img_a_dir or (split_dir / "A")
        self.img_b_dir = img_b_dir or (split_dir / "B")
        self.label_dir = label_dir or (split_dir / "label")

        # 获取所有图像文件名
        self.img_names = self._get_image_names()

        print(f"[{split.upper()}] Loaded {len(self.img_names)} image pairs")


    def _get_image_names(self) -> List[str]:
        if not self.img_a_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.img_a_dir} (expected {self.root_dir}/{self.split}/A or {self.root_dir}/{self.split}/Image1, "
                f"or nested {self.root_dir}/{self.split}/{self.split}/...)"
            )

        stems_a = []
        for ext in EXTS:
            stems_a += [f.stem for f in self.img_a_dir.glob(f"*{ext}")]
        stems_a = sorted(list(set(stems_a)))

        def _exists_any(folder: Path, stem: str) -> bool:
            for ext in EXTS:
                if (folder / f"{stem}{ext}").exists():
                    return True
            return False

        kept = []
        dropped = 0
        for stem in stems_a:
            if not _exists_any(self.img_b_dir, stem):
                dropped += 1
                continue
            if not _exists_any(self.label_dir, stem):
                dropped += 1
                continue
            kept.append(stem)

        if dropped > 0:
            print(f"[{self.split.upper()}] Warning: dropped {dropped} samples missing B/label match.")
        return kept

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> dict:
        img_name = self.img_names[idx]

        img_a_path = _find_existing(self.img_a_dir, img_name)
        img_b_path = _find_existing(self.img_b_dir, img_name)
        label_path = _find_existing(self.label_dir, img_name)

        img_a = _read_rgb_any(img_a_path)
        img_b = _read_rgb_any(img_b_path)
        label = _read_mask_any(label_path)

        # 应用变换
        if self.transform:
            # 将两张图像拼接以保证同步变换
            # albumentations的additional_targets功能
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed['image']
            img_b = transformed['image2']
            label = transformed['mask']
        else:
            # 默认转换为tensor
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()

        return {
            'img_a': img_a,
            'img_b': img_b,
            'label': label,
            'name': img_name
        }


def get_train_transforms(crop_size: int = 256) -> A.Compose:
    """训练集数据增强"""
    transforms_list = []

    # 随机裁剪（从1024裁剪到256）
    if cfg.RANDOM_CROP:
        transforms_list.append(
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0)
        )

    # 几何变换
    if cfg.AUG_HFLIP:
        transforms_list.append(A.HorizontalFlip(p=cfg.AUG_HFLIP_PROB))

    if cfg.AUG_VFLIP:
        transforms_list.append(A.VerticalFlip(p=cfg.AUG_VFLIP_PROB))

    if cfg.AUG_ROTATE:
        transforms_list.append(A.RandomRotate90(p=cfg.AUG_ROTATE_PROB))

    # 颜色变换（对两张图同时应用）
    if cfg.AUG_COLOR_JITTER:
        transforms_list.append(
            A.ColorJitter(
                brightness=cfg.AUG_BRIGHTNESS,
                contrast=cfg.AUG_CONTRAST,
                saturation=cfg.AUG_SATURATION,
                hue=cfg.AUG_HUE,
                p=0.5
            )
        )

    # 高斯噪声
    if cfg.AUG_GAUSSIAN_NOISE:
        try:
            # 新版本albumentations
            transforms_list.append(
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3)
            )
        except TypeError:
            # 旧版本albumentations
            transforms_list.append(
                A.GaussNoise(var_limit=cfg.AUG_NOISE_VAR_LIMIT, p=0.3)
            )

    # 归一化和转换为tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return A.Compose(
        transforms_list,
        additional_targets={'image2': 'image'}  # 确保image2应用相同变换
    )


def get_val_transforms(crop_size: int = 256) -> A.Compose:
    """验证/测试集数据变换（无增强，只裁剪中心）"""
    transforms_list = []

    # 中心裁剪
    if crop_size < cfg.ORIGINAL_SIZE:
        transforms_list.append(
            A.CenterCrop(height=crop_size, width=crop_size)
        )

    # 归一化和转换为tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return A.Compose(
        transforms_list,
        additional_targets={'image2': 'image'}
    )


def get_test_transforms_full() -> A.Compose:
    """测试集变换（完整1024x1024，用于最终评估）"""
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ],
        additional_targets={'image2': 'image'}
    )


def worker_init_fn(worker_id):
    """确保多进程数据加载的随机性"""
    seed = int(np.random.get_state()[1][0]) + worker_id  # 转换为Python int
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def scd_worker_init_fn(worker_id: int):
    """SCD 数据加载器的 worker 初始化（Windows 下必须是顶层函数，才能被 pickle）。"""
    seed = int(getattr(cfg, "SEED", 42)) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证、测试数据加载器"""
    if getattr(cfg, "TASK", "cd") == "scd":
        return create_scd_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # 创建数据集
    train_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='train',
        transform=get_train_transforms(crop_size),
        crop_size=crop_size
    )

    val_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='val',
        transform=get_val_transforms(crop_size),
        crop_size=crop_size
    )

    test_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='test',
        transform=get_val_transforms(crop_size),
        crop_size=crop_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader


SCD_WINTER_RE = re.compile(r"(\d{4})_(\d{4})")


def _resolve_from_tools_base(p: str, repo_root: Path) -> Path:
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return Path("")
    # Patch index CSVs may come from Windows (..\data\...) or Linux (data/...).
    # Normalize to POSIX-style first; Path() will handle it on both OSes.
    p = p.replace("\\", "/")
    # Handle Windows absolute paths when running on Linux, e.g. "D:/Paper/snow_cover/data/...".
    # On POSIX, these are NOT absolute, so we remap them to repo-relative if possible.
    if re.match(r"^[A-Za-z]:/", p):
        low = p.lower()
        # Prefer keeping the portion after the repo folder name if present.
        marker = "/snow_cover/"
        if marker in low:
            tail = p[low.index(marker) + len(marker) :]
            return (repo_root / tail).resolve()
        # Otherwise, try to remap anything under data/ into this repo.
        marker = "/data/"
        if marker in low:
            tail = p[low.index(marker) + len(marker) :]
            return (repo_root / "data" / tail).resolve()
        # Last resort: drop the drive letter and treat as repo-relative.
        p = re.sub(r"^[A-Za-z]:/", "", p)

    path = Path(p)
    if path.is_absolute():
        return path
    # Prefer resolving relative to repo root (Linux-friendly "data/..."),
    # then fall back to resolving relative to repo_root/tools (Windows-generated "..\\data\\...").
    cand_root = (repo_root / path).resolve()
    if cand_root.exists():
        return cand_root
    cand_tools = (repo_root / "tools" / path).resolve()
    return cand_tools


def _nan_stats(arr: np.ndarray):
    """
    arr: (T, H, W) float32 with NaN for invalid.
    Returns mean, std, min, max (H, W) float32 (NaN-safe, fills invalid with 0).
    """
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    amin = np.nanmin(arr, axis=0)
    amax = np.nanmax(arr, axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std), std, 0.0).astype(np.float32)
    amin = np.where(np.isfinite(amin), amin, 0.0).astype(np.float32)
    amax = np.where(np.isfinite(amax), amax, 0.0).astype(np.float32)
    return mean, std, amin, amax


class SnowSCDPatchDataset(Dataset):
    """
    Snow Cover Days (SCD) patch dataset.

    Input (C=11):
      - ERA5 temp stats: mean/std/min/max (4)
      - ERA5 prcp stats: sum/mean/max (3)
      - MODIS static: elev/slope/north/ndvi (4)

    Target:
      - SCD: count of days with snow per pixel, range 0..181
        Computed from MODIS NDSI bands (1..181):
          valid = (ndsi != 255)
          snow  = valid & (ndsi >= cfg.SCD_NDSI_THRESHOLD)
          scd   = snow.sum(axis=0)
    """

    def __init__(self, split: str):
        super().__init__()
        self.split = split
        # PyTorch < 2.3 is generally incompatible with NumPy 2.x for from_numpy()/tensor conversion.
        # Fail fast with a clear message.
        try:
            _ = torch.from_numpy(np.zeros((1,), dtype=np.float32))
        except Exception as e:
            raise RuntimeError(
                "PyTorch cannot interop with your NumPy build (torch.from_numpy failed). "
                "Fix the env by downgrading NumPy to <2 (e.g. `conda install numpy=1.26` or `pip install \"numpy<2\"`), "
                "or upgrade PyTorch to a version built for NumPy 2.x."
            ) from e
        self.repo_root = Path(cfg.PROJECT_ROOT)

        train_csv = Path(getattr(cfg, "SCD_PATCH_INDEX_TRAIN", getattr(cfg, "SCD_PATCH_INDEX", "")))
        val_csv = Path(getattr(cfg, "SCD_PATCH_INDEX_VAL", ""))
        test_csv = Path(getattr(cfg, "SCD_PATCH_INDEX_TEST", ""))
        eval_csv = Path(getattr(cfg, "SCD_PATCH_INDEX_EVAL", getattr(cfg, "SCD_PATCH_INDEX", "")))
        if split == "train":
            self.csv_path = train_csv
        elif split == "val" and val_csv.exists():
            self.csv_path = val_csv
        elif split == "test" and test_csv.exists():
            self.csv_path = test_csv
        else:
            self.csv_path = eval_csv
        if not self.csv_path.exists():
            raise FileNotFoundError(f"SCD_PATCH_INDEX not found: {self.csv_path}")

        rows = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        by_winter: dict[str, list[dict]] = {}
        for r in rows:
            gid = r.get("group_id", "")
            m = SCD_WINTER_RE.search(gid)
            winter = f"{m.group(1)}_{m.group(2)}" if m else "unknown"
            by_winter.setdefault(winter, []).append(r)

        winters = [w for w in sorted(by_winter.keys()) if w != "unknown"]
        val_winters = list(getattr(cfg, "SCD_VAL_WINTERS", []) or [])
        test_winters = list(getattr(cfg, "SCD_TEST_WINTERS", []) or [])

        if not val_winters and not test_winters and len(winters) >= 4:
            test_winters = winters[-2:]
            val_winters = winters[-3:-2]

        if split == "train":
            keep = [w for w in winters if (w not in val_winters and w not in test_winters)]
        elif split == "val":
            keep = val_winters
        elif split == "test":
            keep = test_winters
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples: list[dict] = []
        for w in keep:
            self.samples.extend(by_winter.get(w, []))

        self._ds_cache: dict[str, gdal.Dataset] = {}
        print(f"[SCD/{split}] winters={keep} samples={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _ds(self, path: Path) -> gdal.Dataset:
        key = str(path)
        ds = self._ds_cache.get(key)
        if ds is None:
            ds = gdal.Open(key, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Cannot open raster: {path}")
            self._ds_cache[key] = ds
        return ds

    def __getitem__(self, idx: int) -> dict:
        r = self.samples[idx]
        modis_path = _resolve_from_tools_base(r["modis_path"], self.repo_root)
        era5_temp_path = _resolve_from_tools_base(r["era5_temp_path"], self.repo_root)
        era5_prcp_path = _resolve_from_tools_base(r["era5_prcp_path"], self.repo_root)

        xoff = int(float(r["xoff"]))
        yoff = int(float(r["yoff"]))
        patch = int(float(r.get("patch", cfg.IMAGE_SIZE)))

        ds_m = self._ds(modis_path)
        ds_t = self._ds(era5_temp_path)
        ds_p = self._ds(era5_prcp_path)

        # --- label: SCD from MODIS NDSI ---
        ndsi = ds_m.ReadAsArray(
            xoff, yoff, patch, patch, band_list=list(range(1, 182))
        )
        if ndsi is None:
            raise RuntimeError(f"ReadAsArray failed: {modis_path}")
        ndsi = ndsi.astype(np.float32)
        valid = ndsi != 255.0
        snow = valid & (ndsi >= float(cfg.SCD_NDSI_THRESHOLD))
        scd = snow.sum(axis=0).astype(np.float32)

        # --- static features from MODIS ---
        static = ds_m.ReadAsArray(xoff, yoff, patch, patch, band_list=[363, 364, 365, 366])
        if static is None:
            raise RuntimeError(f"ReadAsArray failed (static): {modis_path}")
        static = static.astype(np.float32)
        elev = static[0]
        slope = static[1] / 100.0
        north = static[2] / 10000.0
        ndvi = static[3] / 10000.0

        elev = np.clip(elev, 0.0, 9000.0) / 9000.0
        slope = np.clip(slope, 0.0, 90.0) / 90.0
        north = (np.clip(north, -1.0, 1.0) + 1.0) / 2.0
        ndvi = np.clip(ndvi, 0.0, 1.0)

        # --- ERA5 features ---
        temp = ds_t.ReadAsArray(xoff, yoff, patch, patch, band_list=list(range(1, 182)))
        prcp = ds_p.ReadAsArray(xoff, yoff, patch, patch, band_list=list(range(1, 182)))
        if temp is None or prcp is None:
            raise RuntimeError(
                f"ReadAsArray failed (era5): temp={era5_temp_path} prcp={era5_prcp_path}"
            )

        temp = temp.astype(np.float32)
        prcp = prcp.astype(np.float32)
        temp[temp <= -9000.0] = np.nan
        prcp[prcp <= -9000.0] = np.nan

        mode = str(getattr(cfg, "SCD_INPUT_MODE", "stats")).lower()
        if mode == "temporal":
            t0, t1 = getattr(cfg, "SCD_TEMP_CLIP", (-40.0, 20.0))
            p0, p1 = getattr(cfg, "SCD_PRCP_CLIP", (0.0, 50.0))

            temp = np.clip(temp, t0, t1)
            prcp = np.clip(prcp, p0, p1)

            # normalize to 0..1, nodata->0
            temp = np.where(np.isfinite(temp), (temp - t0) / max(1e-6, (t1 - t0)), 0.0).astype(
                np.float16
            )
            prcp = np.where(np.isfinite(prcp), (prcp - p0) / max(1e-6, (p1 - p0)), 0.0).astype(
                np.float16
            )

            # pack dynamic as (2*T, H, W) then append static(4)
            dyn = np.concatenate([temp, prcp], axis=0)  # (2T,H,W)
            sta = np.stack([elev, slope, north, ndvi], axis=0).astype(np.float16)
            x = np.concatenate([dyn, sta], axis=0)
        else:
            # stats mode (low memory)
            t_mean, t_std, t_min, t_max = _nan_stats(temp)
            p_mean = np.nanmean(prcp, axis=0)
            p_sum = np.nansum(prcp, axis=0)
            p_max = np.nanmax(prcp, axis=0)

            p_mean = np.where(np.isfinite(p_mean), p_mean, 0.0).astype(np.float32)
            p_sum = np.where(np.isfinite(p_sum), p_sum, 0.0).astype(np.float32)
            p_max = np.where(np.isfinite(p_max), p_max, 0.0).astype(np.float32)

            x = np.stack(
                [
                    t_mean,
                    t_std,
                    t_min,
                    t_max,
                    p_sum,
                    p_mean,
                    p_max,
                    elev,
                    slope,
                    north,
                    ndvi,
                ],
                axis=0,
            ).astype(np.float32)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(scd).unsqueeze(0)

        return {
            "x": x_t,
            "y": y_t,
            "group_id": r.get("group_id", ""),
            "modis_path": str(modis_path),
            "xoff": xoff,
            "yoff": yoff,
        }


def create_scd_dataloaders(batch_size: int = 4, num_workers: int = 2):
    train_ds = SnowSCDPatchDataset(split="train")
    val_ds = SnowSCDPatchDataset(split="val")
    test_ds = SnowSCDPatchDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=scd_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=scd_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=scd_worker_init_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    print("Testing LEVIR-CD Dataset Loading...")
    cfg.display()

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=4,
            num_workers=0,  # 测试时用0
            crop_size=256
        )

        # 测试一个batch
        batch = next(iter(train_loader))
        print(f"\nBatch contents:")
        print(f"  img_a shape: {batch['img_a'].shape}")
        print(f"  img_b shape: {batch['img_b'].shape}")
        print(f"  label shape: {batch['label'].shape}")
        print(f"  label unique values: {torch.unique(batch['label'])}")
        print(f"  names: {batch['name']}")

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")

        print("\nDataset loading test passed!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure LEVIR-CD dataset is placed in data/LEVIR-CD/")
