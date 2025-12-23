# Brain Tumor Detection (MRI) — Keras + Streamlit

Binary classification of brain MRI scans (tumor vs no tumor) with:
- HOG-based MLP baseline (fast, lightweight)
- Optional SVM on HOG features
- Simple CNN on raw pixels with augmentation
- Streamlit UI for uploads and visualization

> **Disclaimer:** For educational/research use only. Not for clinical decisions.

## Dataset
- Kaggle: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Expected layout after download:
  ```
  data/
    yes/   # tumor images
    no/    # non-tumor images
  ```
- Keep large artifacts out of Git (see `.gitignore`): `data/`, `models/`, `reports/`, `venv/`, `__pycache__/`.

## Quickstart
```pwsh
# 1) Create & activate venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Place dataset under data/yes and data/no

# 4) Train HOG+MLP
python train.py --data_dir data --model_dir models --epochs 20 --batch_size 32

# 5) Evaluate
python evaluate.py --data_dir data --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl --output_dir reports

# 6) Predict a single image
python predict.py --image path\to\image.jpg --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl

# 7) Launch Streamlit app
streamlit run app.py
```

## Project Map
```
preprocessing.py     # contour → crop → resize → normalize → HOG
augmentation.py      # ImageDataGenerator configs
model_architectures.py
train.py             # train MLP (default), optional SVM/CNN
evaluate.py          # metrics + plots (confusion matrix, ROC)
predict.py           # CLI prediction
app.py               # Streamlit UI
models/              # saved models & scalers (created after training)
reports/             # evaluation outputs (created after eval)
```

## Detailed Usage
- **Training (MLP baseline)**  
  ```pwsh
  python train.py --data_dir data --model_dir models --epochs 20 --batch_size 32
  ```
  Flags: `--train_svm` (SVM on HOG), `--train_cnn` (CNN with augmentation).

- **Evaluation**  
  ```pwsh
  python evaluate.py --data_dir data --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl --output_dir reports
  ```

- **Prediction (CLI)**  
  ```pwsh
  python predict.py --image path\to\image.jpg --model_path models/best_mlp.h5 --scaler_path models/hog_scaler.pkl
  ```

- **Streamlit app**  
  ```pwsh
  streamlit run app.py
  ```
  Upload an MRI; see original vs preprocessed, prediction, confidence, mask, and HOG visualization.

## Preprocessing Pipeline
1) Contour brain → 2) Crop to largest contour (fallback center crop) → 3) Resize (224×224) → 4) Normalize to [0,1] → 5) HOG features.  
Call `preprocess_image(path)` to get intermediates (mask, cropped, resized, HOG features & visualization).

## Tips
- Use the provided seeds for reproducibility.
- HOG+MLP is fast; CNN may perform better with more data.
- If you see low-confidence outputs, inspect preprocessing visuals (mask/HOG) and consider retraining with the CNN and augmentation.
- To silence the Keras `.h5` warning, save as `.keras` if you prefer.

## Housekeeping
- Keep `data/`, `models/`, `reports/`, and `venv/` out of version control (already in `.gitignore`).
- GPU users: TensorFlow 2.13 on Windows needs CUDA 11.8 + cuDNN 8.6 (see `requirements.txt` note).

