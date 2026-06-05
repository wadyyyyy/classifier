#!/usr/bin/env python3
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


def infer_band_names(band_paths):
    names = []
    for i, p in enumerate(band_paths, start=1):
        stem = Path(p).stem.upper()
        if "B02" in stem:
            names.append("B02")
        elif "B03" in stem:
            names.append("B03")
        elif "B04" in stem:
            names.append("B04")
        elif "B08" in stem:
            names.append("B08")
        else:
            names.append(f"band{i}")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка обучающей выборки из Sentinel-2 и обучающих полигонов"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        required=True,
        help="Пути к каналам (порядок важен: B02 B03 B04 B08)",
    )
    parser.add_argument(
        "--band-names",
        nargs="+",
        default=None,
        help="Имена каналов для столбцов (по умолчанию берутся из имени файла)",
    )
    parser.add_argument(
        "--training",
        required=True,
        help="Векторный файл с полигонами и полем класса (GeoJSON/GeoPackage) ",
    )
    parser.add_argument(
        "--class-field",
        default="class",
        help="Название поля класса в обучающих полигонах",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Выходной CSV с обучающими примерами",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="Минимум пикселей на класс (иначе выводится предупреждение)",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Опциональное ограничение числа пикселей на класс (для баланса)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайной выборки",
    )
    args = parser.parse_args()

    band_paths = args.bands
    band_names = args.band_names or infer_band_names(band_paths)
    if len(band_paths) != len(band_names):
        raise ValueError("Количество путей к каналам должно совпадать с числом имен")

    print("[1/5] Открываю каналы...", flush=True)
    # Открываем каналы и проверяем согласованность
    band_srcs = [rasterio.open(p) for p in band_paths]
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

        print(
            f"[2/5] Растр: {width}x{height}, CRS={crs}",
            flush=True,
        )

        # Читаем обучающие полигоны
        print("[3/5] Читаю обучающие полигоны...", flush=True)
        gdf = gpd.read_file(args.training)
        if args.class_field not in gdf.columns:
            raise ValueError(
                f"Поле '{args.class_field}' не найдено в обучающих полигонах"
            )

        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        print(
            f"[3/5] Всего объектов разметки: {len(gdf)}",
            flush=True,
        )

        # Читаем все каналы в память
        print("[4/5] Читаю каналы в память...", flush=True)
        stack = np.stack([src.read(1) for src in band_srcs], axis=0)
        nodata_vals = [src.nodata for src in band_srcs]

        rng = np.random.default_rng(args.seed)
        samples = []

        for cls_value in sorted(gdf[args.class_field].dropna().unique()):
            subset = gdf[gdf[args.class_field] == cls_value]
            shapes = [
                geom
                for geom in subset.geometry
                if geom is not None and not geom.is_empty
            ]
            if not shapes:
                continue

            print(
                f"[4/5] Класс '{cls_value}': {len(shapes)} полигонов",
                flush=True,
            )

            mask = geometry_mask(
                shapes,
                transform=transform,
                invert=True,
                out_shape=(height, width),
            )

            valid = mask.copy()
            for i, nd in enumerate(nodata_vals):
                if nd is None:
                    continue
                if np.isnan(nd):
                    valid &= ~np.isnan(stack[i])
                else:
                    valid &= stack[i] != nd

            valid &= np.all(np.isfinite(stack), axis=0)
            X = stack[:, valid].T

            if X.shape[0] < args.min_pixels:
                print(
                    f"[WARN] Класс '{cls_value}' содержит всего {X.shape[0]} пикселей"
                )

            if args.max_samples_per_class and X.shape[0] > args.max_samples_per_class:
                idx = rng.choice(X.shape[0], args.max_samples_per_class, replace=False)
                X = X[idx]

            df_cls = pd.DataFrame(X, columns=band_names)
            df_cls["class"] = cls_value
            samples.append(df_cls)

        if not samples:
            raise RuntimeError("Не удалось извлечь пиксели — проверьте полигоны")

        df = pd.concat(samples, ignore_index=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[5/5] Сохранено {len(df)} образцов в {args.output}", flush=True)

    finally:
        for src in band_srcs:
            src.close()


if __name__ == "__main__":
    main()
