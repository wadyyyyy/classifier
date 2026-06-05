#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Объединение нескольких выборок (CSV) в один файл"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Список CSV файлов (samples.csv) от разных сцен",
    )
    parser.add_argument(
        "--output", required=True, help="Куда сохранить объединенный CSV"
    )
    parser.add_argument("--class-field", default="class", help="Имя поля класса")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Не перемешивать строки",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для shuffle")
    args = parser.parse_args()

    frames = []
    base_cols = None

    for path in args.inputs:
        print(f"Читаю {path}...", flush=True)
        df = pd.read_csv(path)
        print(f"  строк: {len(df)}", flush=True)
        if args.class_field not in df.columns:
            raise ValueError(f"В {path} не найдено поле '{args.class_field}'")

        if base_cols is None:
            base_cols = list(df.columns)
        else:
            if set(df.columns) != set(base_cols):
                raise ValueError(
                    f"В {path} другой набор колонок. Ожидалось {base_cols}, получено {list(df.columns)}"
                )
            df = df[base_cols]

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"Всего после объединения: {len(merged)}", flush=True)

    if not args.no_shuffle:
        merged = merged.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Сохранено {len(merged)} строк в {args.output}")


if __name__ == "__main__":
    main()
