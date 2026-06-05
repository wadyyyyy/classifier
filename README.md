# Автоматическое выделение карьеров/отвалов по спектральным характеристикам (Sentinel-2 + SVM)

Этот проект содержит простую реализацию классификации пикселей по спектральным каналам Sentinel‑2 с помощью SVM (scikit‑learn) и чтением растров через rasterio.

## Быстрый старт

1. Установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Подготовить данные (папка `data/`):
- `B02.tif`, `B03.tif`, `B04.tif`, `B08.tif` (10 м)
- `training.geojson` или `training.gpkg` — обучающие полигоны с полем `class` (`water`, `forest`, `quarry`, `urban`)

3. Сформировать обучающую выборку:

```bash
python scripts/prepare_samples.py \
  --bands data/B02.tif data/B03.tif data/B04.tif data/B08.tif \
  --training data/training.gpkg \
  --output data/samples.csv
```

Если сцен несколько, сформируй `samples.csv` для каждой, затем объедини:

```bash
python scripts/merge_samples.py \
  --inputs data/samples_sibai.csv data/samples_scene2.csv data/samples_scene3.csv \
  --output data/samples_all.csv
```

4. Обучить SVM и получить метрики:

```bash
python scripts/train_svm.py \
  --samples data/samples.csv \
  --model models/svm.joblib \
  --report outputs/metrics.txt \
  --confusion outputs/confusion.csv
```

5. Классифицировать весь растр:

```bash
python scripts/classify_raster.py \
  --bands data/B02.tif data/B03.tif data/B04.tif data/B08.tif \
  --model models/svm.joblib \
  --output outputs/class_map.tif
```

## Что внутри

- `scripts/prepare_samples.py` — выборка спектральных значений внутри обучающих полигонов
- `scripts/train_svm.py` — обучение SVM и оценка качества
- `scripts/classify_raster.py` — классификация всех пикселей и запись GeoTIFF
- `docs/report_draft.md` — черновик теории и описания метода для отчёта

## Примечания

- Каналы должны быть в одной проекции/разрешении и иметь одинаковый размер.
- Поле классов в векторе называется `class` (можно изменить флагом `--class-field`).
- Числовые метки классов задаются автоматически: `water=1`, `forest=2`, `quarry=3`, `urban=4` (фиксированный порядок).
