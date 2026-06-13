# 🏥 Disease Prediction from Medical Data — CodeAlpha ML Internship Task 2

A machine learning project that predicts the **possibility of diseases** based on patient data using multiple classification algorithms across three real-world medical datasets.

---

## 📌 Objective
Predict whether a patient has a disease based on clinical features using **SVM, Logistic Regression, Random Forest, and XGBoost** on three different medical datasets.

---

## 📁 Project Structure
```
disease_prediction/
│
├── disease_prediction.py         # Main ML pipeline script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── outputs/
│   ├── datasets/
│   │   ├── heart_disease.csv     # Heart disease dataset (1025 samples)
│   │   ├── diabetes.csv          # Diabetes dataset (768 samples)
│   │   └── breast_cancer.csv     # Breast cancer dataset (569 samples)
│   │
│   ├── plots/
│   │   ├── 01_dataset_overview.png         # Class distribution per dataset
│   │   ├── 02_heart_eda.png                # Heart disease feature distributions
│   │   ├── 03_diabetes_eda.png             # Diabetes feature distributions
│   │   ├── 04_breast_cancer_eda.png        # Breast cancer feature distributions
│   │   ├── 05_correlation_heatmaps.png     # Feature correlations
│   │   ├── 06_heart_disease_model_comparison.png
│   │   ├── 07_diabetes_model_comparison.png
│   │   ├── 08_breast_cancer_model_comparison.png
│   │   ├── 09_roc_curves_all.png           # ROC curves for all models & datasets
│   │   ├── 10_confusion_matrices.png       # 3×4 confusion matrix grid
│   │   ├── 11_feature_importance.png       # RF & XGBoost feature importances
│   │   └── 12_overall_summary.png          # Heatmap summary of all results
│   │
│   └── results/
│       ├── heart_disease_results.csv
│       ├── diabetes_results.csv
│       └── breast_cancer_results.csv
```

---

## 🗂️ Datasets

| Dataset | Samples | Features | Source |
|---|---|---|---|
| Heart Disease | 1025 | 13 | Cleveland Heart Disease (UCI ML) |
| Diabetes | 768 | 8 | Pima Indians Diabetes (UCI ML) |
| Breast Cancer | 569 | 30 | Wisconsin Breast Cancer (UCI ML / sklearn) |

### Heart Disease Features
`age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`, `fasting_blood_sugar`, `rest_ecg`, `max_heart_rate`, `exercise_angina`, `st_depression`, `slope`, `num_vessels`, `thal`

### Diabetes Features
`pregnancies`, `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`, `diabetes_pedigree`, `age`

### Breast Cancer Features
30 features including mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

---

## 🤖 Algorithms Used
- **SVM** — Support Vector Machine with RBF kernel
- **Logistic Regression** — Baseline linear classifier
- **Random Forest** — Ensemble of 100 decision trees
- **XGBoost** — Gradient boosting with 100 estimators

All models use `StandardScaler` preprocessing via sklearn `Pipeline`.

---

## 📊 Results

### 🫀 Heart Disease
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.8585 | 0.8600 | 0.8515 | 0.8557 | 0.9291 |
| Logistic Regression | **0.8780** | **0.8800** | **0.8713** | **0.8756** | **0.9548** 🏆 |
| Random Forest | 0.8244 | 0.8421 | 0.7921 | 0.8163 | 0.9252 |
| XGBoost | 0.8049 | 0.8081 | 0.7921 | 0.8000 | 0.9072 |

### 🩸 Diabetes
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.7792 | 0.8000 | 0.8932 | 0.8440 | 0.8182 |
| Logistic Regression | **0.8052** | **0.8288** | **0.8932** | **0.8598** | **0.8559** 🏆 |
| Random Forest | 0.7922 | 0.8142 | 0.8932 | 0.8519 | 0.8378 |
| XGBoost | 0.7468 | 0.8019 | 0.8252 | 0.8134 | 0.8011 |

### 🎗️ Breast Cancer
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | **0.9960** 🏆 |
| Random Forest | **0.9737** | **1.0000** | **0.9286** | **0.9630** | 0.9929 |
| XGBoost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9940 |

---

## ⚙️ How to Run

```bash
# Clone the repo
git clone https://github.com/Ganeshkumar0911/CodeAlpha-ML-Internship.git
cd disease_prediction

# Install dependencies
pip install -r requirements.txt

# Run the model
python disease_prediction.py
```

---

## 📦 Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
imbalanced-learn
```

---

## 👤 Author
**[Ganesh Kumar S]**
CodeAlpha Machine Learning Internship
