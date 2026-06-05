# Команды запуска (3 территории, RBF SVM + ограничение выборки)

Запускай из корня проекта `proj`.

## 1) Подготовка выборок с ограничением 20k на класс

```bash
python scripts/prepare_samples.py \
  --bands data/sibai/B02.tiff data/sibai/B03.tiff data/sibai/B04.tiff data/sibai/B08.tiff \
  --training data/sibai/training.gpkg \
  --output data/samples_sibai.csv \
  --max-samples-per-class 20000
```

```bash
python scripts/prepare_samples.py \
  --bands data/oskol/B02.tiff data/oskol/B03.tiff data/oskol/B04.tiff data/oskol/B08.tiff \
  --training data/oskol/training.gpkg \
  --output data/samples_oskol.csv \
  --max-samples-per-class 20000
```

```bash
python scripts/prepare_samples.py \
  --bands data/korkino/B02.tiff data/korkino/B03.tiff data/korkino/B04.tiff data/korkino/B08.tiff \
  --training data/korkino/training.gpkg \
  --output data/samples_korkino.csv \
  --max-samples-per-class 20000
```

> Если поле с классом называется не `class`, добавь `--class-field имя_поля`.

## 2) Объединение выборок

```bash
python scripts/merge_samples.py \
  --inputs data/samples_sibai.csv data/samples_oskol.csv data/samples_korkino.csv \
  --output data/samples_all.csv
```

## 3) Обучение SVM (RBF) и метрики

```bash
python scripts/train_svm.py \
  --samples data/samples_all.csv \
  --model models/svm.joblib \
  --report outputs/metrics.txt \
  --confusion outputs/confusion.csv
```

## 4) Классификация каждой территории

```bash
python scripts/classify_raster.py \
  --bands data/sibai/B02.tiff data/sibai/B03.tiff data/sibai/B04.tiff data/sibai/B08.tiff \
  --model models/svm.joblib \
  --output outputs/class_sibai.tif
```

```bash
python scripts/classify_raster.py \
  --bands data/oskol/B02.tiff data/oskol/B03.tiff data/oskol/B04.tiff data/oskol/B08.tiff \
  --model models/svm.joblib \
  --output outputs/class_oskol.tif
```

```bash
python scripts/classify_raster.py \
  --bands data/korkino/B02.tiff data/korkino/B03.tiff data/korkino/B04.tiff data/korkino/B08.tiff \
  --model models/svm.joblib \
  --output outputs/class_korkino.tif
```
