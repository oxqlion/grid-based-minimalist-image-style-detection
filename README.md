# c1_tabular — Minimalist Photo Classifier

Detects **minimalist vs. busy/complex** photographs using tabular ML on hand-crafted,
grid-based composition features (edge density, sharpness, brightness per 3x3 cell,
rule-of-thirds ratios, colorfulness, etc.). Built on the
[Unsplash Research Dataset (Lite)](https://unsplash.com/data). The final model runs
on-device in an iOS app via Core ML.

## Repository layout

```
ml_pipeline/    Feature extraction, clustering (pseudo-labels), supervised training
  aesthetic_minimalist_detection.ipynb              Unsupervised pipeline: download, features, KMeans/GMM/DBSCAN
  supervised_aesthetic_minimalist_detection.ipynb   Supervised models (XGBoost, Random Forest, ...)
  models/                                           Clustering artifacts (PCA, scaler, KMeans, GMM, DBSCAN, config.json)
  data/features/                                    Extracted features and labeled photos (parquet)
  reports/                                          Figures and cluster evaluation
dari_m4_pro/    Final deployment artifacts
  Random_Forest_pipeline.pkl                        Trained Random Forest pipeline
  MinimalistClassifier.mlmodel                      Core ML export
  model_metadata.json, swift_inference_params.json  Feature order and params for Swift inference
  random_forest_parameter_extraction.ipynb          Extracts parameters for the Swift side
swift_app/      iOS app (Xcode project) that runs the Core ML model on camera input
```

## Pipeline

1. **Data** - Unsplash metadata + photos, resized to 256x256.
2. **Features** - 49 features: global (brightness, contrast, sharpness, edge density, saturation,
   colorfulness, texture entropy, hue std), per-cell 3x3 grid (`edge_n`, `sharp_n`, `bright_n`),
   and composition features (`empty_ratio`, `center_ratio`, `thirds_ratio`, balance, ...).
3. **Pseudo-labels** - Clustering (k=2) separates *Minimalist* from *Busy/Complex*.
4. **Supervised model** - Classifiers trained on the labels; Random Forest exported to Core ML.
5. **App** - `swift_app` recomputes the same features on-device and classifies the image.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (deployed) | 0.978 | 0.934 | 0.982 | 0.957 | 0.998 |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn xgboost pyarrow opencv-python pillow matplotlib seaborn jupyter coremltools
```

The raw images are **not** included. Run the first notebook to download them into
`ml_pipeline/data/images/` (ignored by git). Please respect the Unsplash dataset license.

To run the app, open `swift_app/MinimalistClassifier.xcodeproj` in Xcode.

## Notes

- The notebooks and the Swift feature code must stay in sync (same features, same order).
  See `dari_m4_pro/swift_inference_params.json`.
