Automatic Quarry and Waste Dump Detection Using Spectral Features (Sentinel-2 + SVM)

This project provides a simple implementation of pixel-based land cover classification using Sentinel-2 spectral bands and a Support Vector Machine (SVM) model from scikit-learn. Raster data is processed with rasterio.

Quick Start

1. Install Dependencies

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Prepare the Data

Place the following files in the data/ directory:

* B02.tif, B03.tif, B04.tif, B08.tif (10 m Sentinel-2 bands)
* training.geojson or training.gpkg containing training polygons with a class attribute (water, forest, quarry, urban)

3. Generate Training Samples

python scripts/prepare_samples.py \
  --bands data/B02.tif data/B03.tif data/B04.tif data/B08.tif \
  --training data/training.gpkg \
  --output data/samples.csv

If multiple scenes are available, generate a sample file for each scene and merge them:

python scripts/merge_samples.py \
  --inputs data/samples_sibai.csv data/samples_scene2.csv data/samples_scene3.csv \
  --output data/samples_all.csv

4. Train the SVM Model and Evaluate Performance

python scripts/train_svm.py \
  --samples data/samples.csv \
  --model models/svm.joblib \
  --report outputs/metrics.txt \
  --confusion outputs/confusion.csv

5. Classify the Entire Raster

python scripts/classify_raster.py \
  --bands data/B02.tif data/B03.tif data/B04.tif data/B08.tif \
  --model models/svm.joblib \
  --output outputs/class_map.tif

Project Structure

* scripts/prepare_samples.py — extracts spectral values from training polygons
* scripts/train_svm.py — trains the SVM model and evaluates classification performance
* scripts/classify_raster.py — classifies all raster pixels and exports the result as a GeoTIFF
* docs/report_draft.md — draft documentation covering the theoretical background and methodology

Notes

* All raster bands must have the same projection, spatial resolution, and dimensions.
* The class attribute is expected to be named class by default (can be changed with the --class-field argument).
* Numeric class labels are assigned automatically using a fixed mapping:

water  = 1
forest = 2
quarry = 3
urban  = 4
