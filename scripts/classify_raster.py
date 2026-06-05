#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser(
        description="Классификация Sentinel-2 по обученной SVM модели"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        required=True,
        help="Пути к каналам (порядок как в обучении)",
    )
    parser.add_argument("--model", required=True, help="Путь к модели joblib")
    parser.add_argument(
        "--meta",
        default=None,
        help="Путь к JSON с метаданными (по умолчанию <model>.meta.json)",
    )
    parser.add_argument("--output", required=True, help="Выходной GeoTIFF")
    parser.add_argument(
        "--nodata-class",
        type=int,
        default=0,
        help="Класс для пикселей без данных",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    meta_path = args.meta or f"{args.model}.meta.json"
    if Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected_features = meta.get("features")
        if expected_features and len(expected_features) != len(args.bands):
            raise ValueError(
                "Количество каналов не совпадает с числом признаков в модели"
            )
    else:
        meta = None

    band_srcs = [rasterio.open(p) for p in args.bands]
    try:
        ref = band_srcs[0]
        height, width = ref.height, ref.width
        transform = ref.transform
        crs = ref.crs
        for src in band_srcs[1:]:
            if src.height != height or src.width != width:
                raise ValueError("Размеры каналов не совпадают")
            if src.transform != transform:
                raise ValueError("Трансформации каналов не совпадают")
            if src.crs != crs:
                raise ValueError("CRS каналов не совпадает")

        profile = ref.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=args.nodata_class,
            compress="deflate",
        )

        windows = list(ref.block_windows(1))
        total = len(windows)
        step = max(1, total // 10)
        print(f"Классификация: {width}x{height}, окон: {total}", flush=True)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as dst:
            for idx, (_, window) in enumerate(windows, start=1):
                stack = []
                mask = None
                for src in band_srcs:
                    data = src.read(1, window=window, masked=True)
                    stack.append(np.ma.filled(data, np.nan))
                    band_mask = np.ma.getmaskarray(data)
                    mask = band_mask if mask is None else (mask | band_mask)

                arr = np.stack(stack, axis=-1)
                if mask is None:
                    mask = np.zeros((window.height, window.width), dtype=bool)
                valid = ~mask

                out = np.full(valid.shape, args.nodata_class, dtype=np.uint8)
                if valid.any():
                    X = arr[valid].reshape(-1, arr.shape[-1])
                    preds = model.predict(X)
                    out[valid] = preds.astype(np.uint8)

                dst.write(out, 1, window=window)

                if idx % step == 0 or idx == total:
                    pct = (idx / total) * 100
                    print(f"Прогресс: {idx}/{total} ({pct:.1f}%)", flush=True)

        print(f"Классификация сохранена в {args.output}", flush=True)

    finally:
        for src in band_srcs:
            src.close()


if __name__ == "__main__":
    main()
