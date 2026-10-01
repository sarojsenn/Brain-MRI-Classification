# Brain MRI Classification

An educational image-classification project that compares hand-engineered image features, classical machine-learning models, a compact convolutional neural network (CNN), and an experimental Purin/STP CNN on brain MRI images. The four image-level classes are glioma, meningioma, pituitary tumor, and no tumor.

This project is for research and learning only. It is not a medically validated diagnostic system, and its results are not clinical performance estimates.

## Workflow

The workflow in [`brain_mri_classification.ipynb`](brain_mri_classification.ipynb) includes:

1. Inspecting the dataset structure, class counts, image sizes, and file readability.
2. Comparing image preprocessing options, including grayscale conversion, resizing, normalization, Gaussian blur, and Otsu thresholding.
3. Extracting intensity, histogram, shape, edge, gradient, and texture features from resized grayscale images.
4. Comparing a DummyClassifier, Logistic Regression, SVM, Random Forest, and Gradient Boosting classifier using a stratified validation split.
5. Evaluating the selected classical model and compact CNN on the held-out test set.
6. Training an experimental Purin/STP CNN on grayscale images, reporting validation metrics, and generating a Grad-CAM visualization.
7. Saving models and feature metadata and providing single-image prediction functions.

## Dataset Layout

The notebook expects a dataset root containing `train` and `test` directories. Each split contains one directory per class, with supported image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, or `.tiff`).

```text
data/
  train/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
  test/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
```

The notebook searches for `data` in the current workspace and a `brisc2025/classification_task` directory, or can be configured with the dataset's actual location in its setup cell.

## Requirements and Running

Use a Python environment with Jupyter Notebook and the libraries imported by the notebook: NumPy, Pandas, Matplotlib, Seaborn, OpenCV (`opencv-python`), Pillow, scikit-learn, and joblib. TensorFlow is needed for the CNN cells. Run the notebook cells in order. The Purin/STP model also imports the custom layers from [`purin_layers.py`](purin_layers.py), which must remain alongside the notebook.

## Results Reported in the Notebook

### Existing held-out test baselines

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Engineered features + Random Forest | 0.9170 | 0.9195 | 0.9153 |
| Compact CNN baseline | 0.5510 | 0.5713 | 0.5400 |

### Purin/STP validation run

The notebook reports results on a 998-image validation subset from the training folder. These are validation results, not held-out test results, and should not be compared directly with the baseline test scores above.

| Metric | Result |
|---|---:|
| Accuracy | 82.26% |
| Macro precision | 83.05% |
| Macro recall | 83.07% |
| Macro F1 | 80.93% |
| Macro ROC-AUC (OvR) | 93.01% |
| Validation loss | 1.2080 |

The notebook identifies meningioma recall as the main weakness in this Purin/STP run. Metrics are dataset-specific; external clinical performance is unknown.

## Generated Artifacts

The notebook saves engineered feature tables, model files, label and feature metadata, Purin/STP metrics, and a PDF report under `outputs/`. The Purin/STP custom layers are defined in `purin_layers.py` for model reloading. The notebook provides `predict_single_image` for the engineered-feature classifier and `predict_purin_image` for the Purin/STP model.

## Limitations

- This is image-level classification, not tumor segmentation.
- Threshold-derived shape features describe image structure; they are not validated tumor masks.
- Grad-CAM and feature importance are approximate model explanations, not clinical evidence.
- The reported scores do not establish generalization to external clinical data and are not calibrated medical confidence estimates.