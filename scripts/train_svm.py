#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser(
        description="Обучение SVM на спектральных выборках"
    )
    parser.add_argument("--samples", required=True, help="CSV с обучающими примерами")
    parser.add_argument("--class-field", default="class", help="Поле с классом")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Список признаков (по умолчанию все столбцы, кроме class/class_id)",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Доля теста")
    parser.add_argument("--random-state", type=int, default=42, help="Seed")
    parser.add_argument("--kernel", default="rbf", help="Ядро SVM")
    parser.add_argument("--C", type=float, default=10.0, help="Параметр C")
    parser.add_argument("--gamma", default="scale", help="Параметр gamma")
    parser.add_argument("--model", required=True, help="Путь для сохранения модели")
    parser.add_argument(
        "--meta",
        default=None,
        help="Путь для JSON с метаданными (по умолчанию рядом с моделью)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Путь для текстового отчета с метриками",
    )
    parser.add_argument(
        "--confusion",
        default=None,
        help="Путь для CSV с confusion matrix",
    )
    args = parser.parse_args()

    print("[1/4] Читаю выборку...", flush=True)
    df = pd.read_csv(args.samples)
    if args.class_field not in df.columns:
        raise ValueError(f"Поле '{args.class_field}' не найдено в CSV")

    if args.features:
        feature_cols = args.features
    else:
        feature_cols = [
            c for c in df.columns if c not in {args.class_field, "class_id"}
        ]

    print(f"[2/4] Признаки: {feature_cols}", flush=True)
    print(
        "[2/4] Распределение классов:\n"
        + df[args.class_field].value_counts().to_string(),
        flush=True,
    )

    X = df[feature_cols].values
    raw_classes = df[args.class_field].dropna().unique().tolist()
    preferred_order = ["water", "forest", "quarry", "urban"]
    class_names = [c for c in preferred_order if c in raw_classes]
    class_names += sorted([c for c in raw_classes if c not in preferred_order])
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    y = df[args.class_field].map(class_to_id).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(
        f"[3/4] Train: {len(X_train)} | Test: {len(X_test)}",
        flush=True,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(kernel=args.kernel, C=args.C, gamma=args.gamma),
            ),
        ]
    )

    print("[4/4] Обучаю модель...", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(class_to_id.values()))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(class_to_id.values()),
        target_names=class_names,
    )

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)

    meta_path = args.meta or f"{args.model}.meta.json"
    meta = {
        "class_names": class_names,
        "class_to_id": class_to_id,
        "id_to_class": {str(v): k for k, v in class_to_id.items()},
        "features": feature_cols,
        "model": "SVM",
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
    }
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(cm))
            f.write("\n\nClassification report:\n")
            f.write(report)

    if args.confusion:
        Path(args.confusion).parent.mkdir(parents=True, exist_ok=True)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(args.confusion)


if __name__ == "__main__":
    main()
