# Brain Tumor Detection (MRI, Keras)

Binary classifier for detecting brain tumors from MRI images with a shared preprocessing pipeline and Streamlit app.

## Project Structure
```
data/                # yes/ and no/ folders with MRI images
models/              # saved models and scalers (created after training)
preprocessing.py     # contour→crop→resize→normalize→HOG pipeline
augmentation.py      # Keras ImageDataGenerator helpers
model_architectures.py
train.py             # training entrypoint (MLP baseline, optional CNN/SVM)
evaluate.py          # metrics + plots (confusion matrix, ROC)
predict.py           # CLI single-image prediction (HOG+MLP)
app.py               # Streamlit UI for uploads and predictions
requirements.txt
```

## Setup
1. Create a virtual env and install deps:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Ensure dataset is in `data/yes` and `data/no` (already provided).

## Preprocessing Pipeline
Order is fixed: **contour → crop → resize (224×224) → normalize (0-1) → HOG**. Functions live in `preprocessing.py`. Use `preprocess_image(path)` to get all intermediates (mask, cropped, resized, HOG features).

## Training
Baseline MLP on HOG features:
```
python train.py --data_dir data --model_dir models --epochs 20 --batch_size 32
```
Optional flags:
- `--train_svm` to fit an SVM on HOG.
- `--train_cnn` to fit a small CNN using augmentation.

Artifacts saved to `models/`: `best_mlp.h5`, `final_mlp.h5`, `hog_scaler.pkl`, metrics JSON; similarly for SVM/CNN if enabled.

## Evaluation
Generate metrics + plots (confusion matrix, ROC-AUC):
```
python evaluate.py --data_dir data --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl --output_dir reports
```
Outputs go to `reports/`.

## Prediction (CLI)
```
python predict.py --image path/to/mri.jpg --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl
```
Prints label and confidence.

## Streamlit App
1. Train and ensure `models/best_mlp.h5` and `models/hog_scaler.pkl` exist.
2. Run:
   ```
   streamlit run app.py
   ```
3. Upload an MRI image; the app shows original vs preprocessed, prediction, confidence, mask, and HOG visualization.

## Notes and Recommendations
- Set seeds are baked in for reproducibility.
- Augmentation: rotation/shift/zoom/horizontal flip via `ImageDataGenerator`.
- Callbacks used: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
- Aim for 80/10/10 or 70/15/15 splits; current defaults are close (train/val/test via train.py).
- For medical contexts, prioritize recall; adjust decision threshold if needed.
- Include the provided disclaimer: results are for educational/research purposes only.

