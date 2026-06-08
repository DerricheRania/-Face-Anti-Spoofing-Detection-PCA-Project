# 🛡️ Face Anti-Spoofing Detection : ACP / PCA Project
---

## 📌 Overview

This project tackles **Face Anti-Spoofing** the problem of automatically distinguishing a **real human face** from a **fake one** (printed photo, replayed video on a screen, or mask). Face anti-spoofing is a critical security layer in modern biometric systems such as phone unlocking, access control, and biometric payment.

The project is split into **two complementary approaches**, both leveraging **PCA (Principal Component Analysis)** for dimensionality reduction:

| | Approach | Dataset | Method |
|---|---|---|---|
| **Part 1** | Deep feature extraction + PCA + classical classifiers | Roboflow Face Anti-Spoofing | EfficientNet-B4 → PCA → SVM / KNN / Logistic Regression |
| **Part 2** | Raw pixel PCA + 5 classifiers | LCC_FASD | Raw pixels → PCA → SVM / RF / LR / GB / MLP |

---

## 📁 Project Structure

```
anti-spoofing-pca/
│
├── 📓 NB_ProjetANADON_ACP_CORRIGE_V3.ipynb   ← Part 1: EfficientNet-B4 + PCA notebook
│
├── 🐍 main.py                                 ← Part 2: Main entry point (run this)
├── 🐍 one_load_and_explore.py                 ← Step 1: Dataset loading & exploration
├── 🐍 two_preprocessing.py                    ← Step 2: Normalization & train/test split
├── 🐍 three_pca.py                            ← Step 3: PCA application & all visualizations
├── 🐍 four_classification.py                  ← Step 4: Training 5 classifiers
├── 🐍 five_evaluation.py                      ← Step 5: Evaluation, ROC, confusion matrix
├── 🐍 Test_image.py                           ← Standalone script to test a single image
│
├── 📄 rapport_anti_spoofing.pdf               ← Full project report
│
├── 📂 LCC_FASD/
│   ├── real/                                  ← 2 254 real face images
│   └── spoof/                                 ← 7 046 spoofed face images
│
└── 📂 outputs/  (generated after running)
    ├── output_1_exemples.png
    ├── output_2_distribution.png
    ├── output_3a_correlation_matrix.png
    ├── output_3b_eigenvalues.png
    ├── output_3d_eigenvectors.png
    ├── output_3_variance_pca.png
    ├── output_4_eigenfaces.png
    ├── output_3g_pca_axes_pairs.png
    ├── output_5_pca_2d.png
    ├── output_5b_pca_3d.png
    ├── output_5c_reconstruction.png
    ├── output_5d_pc_distributions.png
    ├── output_6_evaluation.png
    ├── output_7_comparaison.png
    ├── pca_model.pkl
    ├── scaler.pkl
    └── *_model.pkl  (one per classifier)
```

---

## 🧪 Part 1 : EfficientNet-B4 + PCA + Classifiers (Notebook)

**File:** `NB_ProjetANADON_ACP_CORRIGE_V3.ipynb`

### Dataset
The **Roboflow Face Anti-Spoofing** dataset, annotated in YOLOv8 format (class 0 = real, class 1 = spoof), split into train / valid / test sets.

### Pipeline

```
Images → EfficientNet-B4 (feature extractor) → 1792-dim vectors
       → StandardScaler → PCA (300 components, 92.82% variance)
       → SVM (RBF) / KNN (k=7) / Logistic Regression
```

**Step 1 : EDA:** Count images per class, visualize distribution, display sample images.

**Step 2 : Feature Extraction with EfficientNet-B4:** The last classification layer is removed and the network is used as a feature extractor. Each image (resized to 224×224, normalized with ImageNet statistics) produces a 1792-dimensional feature vector encoding deep texture and contour information.

**Step 3 : PCA:** Features are standardized then reduced. 300 components are chosen to cover 92.82% of total variance. Visualizations include the correlation matrix (before and after PCA), eigenvalue scree plot, 2D/3D projections, and a biplot.

**Step 4 : Classification & Evaluation:**

| Classifier | Test Accuracy | CV-5 Accuracy | AUC-ROC |
|---|---|---|---|
| **SVM (RBF)** | 94.85% | **99.80%** | **0.993** |
| KNN (k=7) | **95.88%** | 95.78% | 0.978 |
| Logistic Regression | 91.75% | 99.47% | 0.971 |

> The SVM with RBF kernel is the best overall, combining high AUC-ROC (0.993) with the most stable cross-validation accuracy.

---

## 🧪 Part 2 : Raw Pixel PCA + 5 Classifiers (Python Scripts)

**Entry point:** `main.py`

### Dataset : LCC_FASD

| Category | Images | Percentage |
|---|---|---|
| Real | 2 254 | 50% |
| Fake (randomly sampled) | 2 254 | 50% |
| **Total** | **4 508** | **100%** |

Images are loaded in grayscale and resized to 64×64 pixels, producing a 4096-dimensional vector per image. A random subset of 2 254 fake images is selected to perfectly balance the two classes.

### Pipeline

```
Images (64×64 grayscale) → Normalize [0,1] → Train/Test split (80/20)
                         → StandardScaler → PCA (50 components, 88.2% variance)
                         → 5 Classifiers → Evaluation
```

---

### Step-by-Step Description

#### Step 1 : Loading & Exploration (`one_load_and_explore.py`)
- Loads images from `LCC_FASD/real/` and `LCC_FASD/spoof/`
- Converts to grayscale, resizes to 64×64
- Randomly samples the fake images to balance classes (seed=42)
- Displays 5 examples per class side by side → `output_1_exemples.png`

#### Step 2 : Preprocessing (`two_preprocessing.py`)
- **Normalization:** pixel values scaled from [0, 255] to [0, 1]
- **Stratified split:** 80% train (3 606 images) / 20% test (902 images)
- Class distribution plot → `output_2_distribution.png`

#### Step 3 : PCA (`three_pca.py`)
PCA is applied **directly on raw pixel vectors** (4096 dimensions → 50 components).

> ⚠️ **Important:** StandardScaler is fit only on the training set, then applied to both train and test to avoid data leakage.

The following visualizations are generated:

| Output file | Description |
|---|---|
| `output_3a_correlation_matrix.png` | Correlation matrix of the first 40 pixels — shows high redundancy before PCA |
| `output_3b_eigenvalues.png` | Scree plot of eigenvalues (λ) + cumulative variance curve |
| `output_3d_eigenvectors.png` | First 6 eigenvectors displayed as 64×64 images (red = positive, blue = negative) |
| `output_3_variance_pca.png` | Elbow curve showing how many components are needed for 80/90/95/99% variance |
| `output_4_eigenfaces.png` | First 15 principal components displayed as "eigenfaces" |
| `output_3g_pca_axes_pairs.png` | Pairwise projections: PC1×PC2, PC1×PC3, PC2×PC3 |
| `output_5_pca_2d.png` | 2D scatter of training data in PC1 vs PC2 space |
| `output_5b_pca_3d.png` | 3D scatter in PC1/PC2/PC3 space |
| `output_5c_reconstruction.png` | Image reconstruction quality at 5 / 10 / 20 / 50 components |
| `output_5d_pc_distributions.png` | Score distribution per class on PC1 and PC2 |

**Key PCA results:**
- PC1 (21.62% variance) and PC2 (17.34% variance) together capture nearly 39% of total information
- 21 components suffice to reach 80% variance; 50 components cover 88.2%
- The correlation matrix reveals strong spatial redundancy between neighboring pixels — exactly what PCA removes

#### Step 4 : Training Classifiers (`four_classification.py`)

Five classifiers are trained on the 50 PCA components:

| Classifier | Train Accuracy | Description |
|---|---|---|
| SVM (RBF) | 97.9% | Non-linear maximum-margin classifier, `C=10`, `gamma='scale'` |
| Random Forest | 100% | 200 decision trees (bagging), likely overfitting |
| Logistic Regression | 72.5% | Linear probabilistic baseline |
| Gradient Boosting | 99.4% | Sequential boosting, 200 estimators, `lr=0.1`, `depth=4` |
| MLP Neural Network | 98.7% | Dense network (256→128), ReLU, early stopping |

All trained models are saved as `.pkl` files for reuse.

#### Step 5 : Evaluation (`five_evaluation.py`)

All 5 models are evaluated on the 902 test images (451 real, 451 fake).

| Classifier | Test Accuracy | F1-score | AUC-ROC | Rank |
|---|---|---|---|---|
| **SVM (RBF)** | **92.68%** | **0.93** | **0.974** | 🥇 1st |
| MLP Neural Network | 91.80% | 0.92 | 0.964 | 🥈 2nd |
| Gradient Boosting | 89.58% | 0.90 | 0.958 | 🥉 3rd |
| Random Forest | 88.91% | 0.89 | 0.955 | 4th |
| Logistic Regression | 72.06% | 0.72 | 0.808 | 5th |

Generated output files:
- `output_6_evaluation.png` — per-model confusion matrix + ROC curve + class metrics
- `output_7_comparaison.png` — comparative bar charts (accuracy & AUC) + all ROC curves overlaid

---

## 🚀 How to Run (Part 2)

### Prerequisites

```bash
pip install numpy matplotlib seaborn scikit-learn Pillow tqdm joblib
```

### Dataset Setup

Place the LCC_FASD dataset in the project root:
```
LCC_FASD/
├── real/      ← .jpg / .png images of real faces
└── spoof/     ← .jpg / .png images of spoofed faces
```

### Run the full pipeline

```bash
python main.py
```

This runs all 5 steps sequentially and saves all output figures and model files to the project directory.

### Test a single image

Edit `Test_image.py` to set your image path and desired model, then run:

```bash
python Test_image.py
```

The script loads the saved `scaler.pkl`, `pca_model.pkl`, and the chosen classifier, and displays the prediction with confidence.

---

## ⚙️ Configuration (Part 2)

All tunable parameters are at the top of `main.py`:

```python
DATA_DIR      = "LCC_FASD"      # Root dataset directory
IMG_SIZE      = (64, 64)         # Target image size
N_COMPONENTS  = 50               # Number of PCA components to keep
TEST_SIZE     = 0.2              # Fraction of data used for testing
RANDOM_STATE  = 42               # Reproducibility seed
```

---

## 🔍 Key Findings

- **SVM with RBF kernel is the best classifier in both approaches.** It defines non-linear decision boundaries in the PCA-reduced space that effectively separate real from fake faces.
- **PCA is essential in both pipelines:** it removes noise and redundancy, decorrelates features, and significantly speeds up classifier training.
- **Random Forest overfits:** 100% train accuracy vs. 88.91% test accuracy shows it memorized the training set rather than generalizing.
- **Logistic Regression underperforms** because its linear decision boundary is insufficient for this problem, even in the reduced PCA space.
- **Testing on out-of-distribution images** (a dog photo) always produces a SPOOF prediction with high confidence — the model has no "unknown" class and will always assign one of the two labels.

---

## 🧠 Why PCA?

| Reason | Explanation |
|---|---|
| **Dimensionality reduction** | Raw pixels: 4096 dimensions. After PCA: 50 dimensions (88.2% variance preserved) |
| **Noise removal** | Low-variance components (mostly noise) are discarded |
| **Decorrelation** | PCA components are orthogonal — no redundant information |
| **Speed** | Classifiers train much faster on 50 dimensions than 4096 |
| **Interpretability** | Eigenfaces visually show what "directions of variation" the model has learned |

---

## 📊 Comparison: Part 1 vs Part 2

| Aspect | Part 1 (EfficientNet + PCA) | Part 2 (Raw Pixels + PCA) |
|---|---|---|
| Input features | 1792 (deep CNN features) | 4096 (raw pixels) |
| PCA components | 300 (92.82% variance) | 50 (88.2% variance) |
| Best classifier | SVM (AUC 0.993) | SVM (AUC 0.974) |
| Best test accuracy | 95.88% (KNN) | 92.68% (SVM) |
| GPU required | Yes (EfficientNet inference) | No |
| Approach complexity | Higher | Lower |

Both approaches confirm that **SVM with RBF kernel** is the most reliable classifier for this problem. Using a pre-trained deep network (Part 1) gives a small but consistent accuracy boost at the cost of needing more compute resources.

---

## 📚 References

- LCC_FASD Dataset: Local Color Contrast Face Anti-Spoofing Dataset
- Roboflow Face Anti-Spoofing Dataset
- EfficientNet-B4: Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- Scikit-learn documentation: https://scikit-learn.org
