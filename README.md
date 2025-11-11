# Multimodal Brain Tumor Segmentation with 3D U-Net

Brief project to preprocess BraTS 2020 MRI data and train a 3D U-Net segmentation model.

---

## Repository layout
- src/
  - exploratory_analsys.py    # NIfTI -> .npy preprocessing (3 channels + mask)
  - dataloader.py             # PyTorch Dataset / utility to find unmatched files
  - train_data.py             # Training loop, metrics tracking and plots saving
  - image_segmentation_with_u_net.py  # UNet model definition
- dataset/
  - BraTS2020_TrainingData/... (generated / saved .npy files)
- README.md

---

## Dataset
Original dataset (downloaded externally) path used in this repo:
/Users/asminanassar/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3

Preprocessed files are placed under:
dataset/BraTS2020_TrainingData/input_data_3channels/images
dataset/BraTS2020_TrainingData/input_data_3channels/masks

Note: exploratory_analsys.py looks for subject NIfTI files, normalizes each volume, stacks modalities into 3-channel volumes and saves .npy arrays.

---

## Requirements / Environment
Create and activate a Python environment (example using virtualenv or conda). Example minimal packages:
- python 3.8+ / 3.9
- numpy
- nibabel
- matplotlib
- scikit-learn
- tensorflow (for to_categorical) — only used in preprocessing
- torch (PyTorch)
- segmentation_models_3D (or the version used)
Install with pip or conda as appropriate. GPU drivers and CUDA for PyTorch if available.

---

## Preprocessing
1. Place raw BraTS NIfTI files in your local folder (see dataset path above).
2. Run the preprocessing script to generate .npy files:
   python src/exploratory_analsys.py
   - This script finds complete subjects (t2/t1ce/flair/seg), min-max normalizes per-volume, crops to the ROI, and saves images/masks as .npy.
   - It skips subjects missing any modality and logs skipped subjects.

Notes:
- The script preserves per-volume min-max normalization behavior.
- If you want to adjust the crop or thresholds, edit combine_slices or the useful-volume threshold.

---

## Finding unmatched images
If you have mismatched .npy counts (images without masks), run the dataloader utility:
python src/dataloader.py
This prints counts and example unmatched image files in:
dataset/BraTS2020_TrainingData/input_data_3channels/

---

## Training
- The training script entrypoint is `src/train_data.py`.
- It uses a combined Dice + Categorical Focal loss (callable) and computes:
  - per-epoch train/val loss
  - accuracy
  - mean IoU (per-epoch)
- Plots and numeric histories are saved in `training_plots/` by default:
  - loss_curve.png, accuracy_curve.png, iou_curve.png
  - train_losses.npy, val_losses.npy, train_accs.npy, val_accs.npy, train_ious.npy, val_ious.npy

Run training (example):
python -c "from src.train_data import training, model, train_loader; training(model, train_loader, None, optimizer, num_epochs=10)"

Notes:
- Provide a validation DataLoader to compute validation metrics.
- For GPU: ensure PyTorch detects CUDA; training moves model and data to `cuda` automatically when available.

---

## Outputs
- Saved .npy image/mask pairs in dataset/.../input_data_3channels/
- Training plots and numeric histories in `training_plots/` (or custom save_dir)
- Modify `train_data.py` to checkpoint models if you need checkpoints.

---

## Troubleshooting
- IndexError in preprocessing usually means lists of modality files were mismatched; the script now only processes subjects with all modalities and logs skipped subjects.
- If DataLoader raises file-not-found errors, run `python src/dataloader.py` to list unmatched files and inspect filenames.
- Warning about LibreSSL / OpenSSL from urllib3 is informational; update your system OpenSSL if needed but it does not stop the pipeline.

---

## Tips
- Standardize filenames when saving .npy to ensure dataloader can match image/mask pairs easily (e.g., `image_###.npy` and `mask_###.npy`).
- If you want a global MinMaxScaler across subjects, compute min/max on a representative sample and apply that fixed range — current code uses per-volume min-max normalization.