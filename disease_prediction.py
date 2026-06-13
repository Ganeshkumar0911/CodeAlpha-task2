# ============================================================
# DISEASE PREDICTION FROM MEDICAL DATA
# CodeAlpha Machine Learning Internship - Task 2
# ============================================================
# Datasets : Heart Disease, Diabetes, Breast Cancer (UCI ML)
# Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import os

os.makedirs("outputs/plots",   exist_ok=True)
os.makedirs("outputs/datasets", exist_ok=True)
os.makedirs("outputs/results",  exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
PALETTE = {'Positive': '#E91E63', 'Negative': '#2196F3'}

print("=" * 65)
print("   DISEASE PREDICTION FROM MEDICAL DATA")
print("   CodeAlpha ML Internship — Task 2")
print("=" * 65)


# ============================================================
# 1.  DATASETS
# ============================================================

np.random.seed(42)

# ── 1A. HEART DISEASE (Cleveland-style, UCI) ─────────────────
def make_heart_disease(n=1025):
    age         = np.random.randint(29, 77, n)
    sex         = np.random.choice([0, 1], n, p=[0.32, 0.68])
    cp          = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps    = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    chol        = np.random.normal(246, 52, n).clip(126, 564).astype(int)
    fbs         = np.random.choice([0, 1], n, p=[0.85, 0.15])
    restecg     = np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
    thalach     = np.random.normal(149, 23, n).clip(71, 202).astype(int)
    exang       = np.random.choice([0, 1], n, p=[0.67, 0.33])
    oldpeak     = np.random.exponential(1.1, n).clip(0, 6.2).round(1)
    slope       = np.random.choice([0, 1, 2], n, p=[0.21, 0.47, 0.32])
    ca          = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal        = np.random.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])

    score = (
        (age - 29) / 48 * 0.15
        + sex * 0.10
        + cp * 0.12
        + (trestbps - 94) / 106 * 0.08
        + exang * 0.12
        + oldpeak / 6.2 * 0.14
        + ca / 3 * 0.16
        + (3 - slope) / 2 * 0.08
        - (thalach - 71) / 131 * 0.10
        + np.random.normal(0, 0.08, n)
    )
    target = (score > score.mean()).astype(int)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'chest_pain_type': cp,
        'resting_bp': trestbps, 'cholesterol': chol,
        'fasting_blood_sugar': fbs, 'rest_ecg': restecg,
        'max_heart_rate': thalach, 'exercise_angina': exang,
        'st_depression': oldpeak, 'slope': slope,
        'num_vessels': ca, 'thal': thal, 'target': target
    })
    return df

# ── 1B. DIABETES (Pima Indians-style, UCI) ───────────────────
def make_diabetes(n=768):
    pregnancies = np.random.poisson(3.8, n).clip(0, 17)
    glucose     = np.random.normal(121, 32, n).clip(44, 199).astype(int)
    bp          = np.random.normal(69, 19, n).clip(0, 122).astype(int)
    skin        = np.random.normal(21, 16, n).clip(0, 99).astype(int)
    insulin     = np.random.exponential(80, n).clip(0, 846).astype(int)
    bmi         = np.random.normal(32, 8, n).clip(18, 67).round(1)
    dpf         = np.random.exponential(0.47, n).clip(0.08, 2.42).round(3)
    age         = np.random.randint(21, 81, n)

    score = (
        (glucose - 44) / 155 * 0.30
        + bmi / 67 * 0.20
        + dpf / 2.42 * 0.15
        + (age - 21) / 60 * 0.15
        + pregnancies / 17 * 0.10
        + np.random.normal(0, 0.08, n)
    )
    outcome = (score > score.mean() - 0.05).astype(int)

    df = pd.DataFrame({
        'pregnancies': pregnancies, 'glucose': glucose,
        'blood_pressure': bp, 'skin_thickness': skin,
        'insulin': insulin, 'bmi': bmi,
        'diabetes_pedigree': dpf, 'age': age, 'outcome': outcome
    })
    return df

# ── 1C. BREAST CANCER (real UCI data via sklearn) ────────────
def load_cancer():
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    df['diagnosis'] = bc.target          # 1 = malignant, 0 = benign  (sklearn encodes 0=malignant, 1=benign — we flip)
    df['diagnosis'] = 1 - df['diagnosis']  # 1 = malignant
    return df

heart_df   = make_heart_disease()
diabetes_df = make_diabetes()
cancer_df  = load_cancer()

heart_df.to_csv('outputs/datasets/heart_disease.csv',   index=False)
diabetes_df.to_csv('outputs/datasets/diabetes.csv',      index=False)
cancer_df.to_csv('outputs/datasets/breast_cancer.csv',  index=False)

datasets = {
    'Heart Disease':   (heart_df,   'target',    'Heart Disease'),
    'Diabetes':        (diabetes_df, 'outcome',   'Diabetes'),
    'Breast Cancer':   (cancer_df,  'diagnosis', 'Malignant')
}

print("\n📊 DATASETS LOADED")
print(f"   Heart Disease  : {heart_df.shape[0]} samples, {heart_df.shape[1]-1} features")
print(f"   Diabetes       : {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features")
print(f"   Breast Cancer  : {cancer_df.shape[0]} samples, {cancer_df.shape[1]-1} features")


# ============================================================
# 2.  EDA PLOTS
# ============================================================

# ── Plot 1: Dataset Overview ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Dataset Overview — Class Distribution', fontsize=15, fontweight='bold')

info = [
    (heart_df,   'target',    ['No Disease', 'Disease'],      '#2196F3'),
    (diabetes_df,'outcome',   ['Non-Diabetic', 'Diabetic'],   '#4CAF50'),
    (cancer_df,  'diagnosis', ['Benign', 'Malignant'],        '#E91E63'),
]
titles = ['Heart Disease', 'Diabetes', 'Breast Cancer']

for ax, (df, col, labels, color), title in zip(axes, info, titles):
    counts = df[col].value_counts().sort_index()
    bars = ax.bar(labels, counts.values,
                  color=[f'{color}55', color], edgecolor='white', linewidth=1.5, width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f'{v}\n({v/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(counts.values)*1.2)

plt.tight_layout()
plt.savefig('outputs/plots/01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Plot 01 saved: Dataset Overview")

# ── Plot 2: Heart Disease EDA ────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Heart Disease — Feature Distributions by Class', fontsize=14, fontweight='bold')
heart_feats = ['age','cholesterol','resting_bp','max_heart_rate',
               'st_depression','num_vessels','chest_pain_type','exercise_angina']
for ax, feat in zip(axes.flatten(), heart_feats):
    for cls, clr, lbl in [(0,'#2196F3','No Disease'),(1,'#E91E63','Disease')]:
        data = heart_df[heart_df['target']==cls][feat]
        ax.hist(data, bins=20, alpha=0.6, color=clr, label=lbl)
    ax.set_title(feat.replace('_',' ').title(), fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/02_heart_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 02 saved: Heart Disease EDA")

# ── Plot 3: Diabetes EDA ─────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Diabetes — Feature Distributions by Class', fontsize=14, fontweight='bold')
diab_feats = ['glucose','bmi','age','blood_pressure',
              'insulin','skin_thickness','pregnancies','diabetes_pedigree']
for ax, feat in zip(axes.flatten(), diab_feats):
    for cls, clr, lbl in [(0,'#4CAF50','Non-Diabetic'),(1,'#E91E63','Diabetic')]:
        data = diabetes_df[diabetes_df['outcome']==cls][feat]
        ax.hist(data, bins=20, alpha=0.6, color=clr, label=lbl)
    ax.set_title(feat.replace('_',' ').title(), fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/03_diabetes_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 03 saved: Diabetes EDA")

# ── Plot 4: Breast Cancer EDA ────────────────────────────────
cancer_feats = [c for c in cancer_df.columns if c != 'diagnosis'][:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Breast Cancer — Feature Distributions by Class', fontsize=14, fontweight='bold')
for ax, feat in zip(axes.flatten(), cancer_feats):
    for cls, clr, lbl in [(0,'#2196F3','Benign'),(1,'#E91E63','Malignant')]:
        data = cancer_df[cancer_df['diagnosis']==cls][feat]
        ax.hist(data, bins=20, alpha=0.6, color=clr, label=lbl)
    ax.set_title(feat.replace('_',' ').title(), fontsize=8, fontweight='bold')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/04_breast_cancer_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 04 saved: Breast Cancer EDA")

# ── Plot 5: Correlation Heatmaps ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Feature Correlation Heatmaps', fontsize=14, fontweight='bold')
for ax, (name, (df, target, _)) in zip(axes, datasets.items()):
    short_df = df.copy()
    cols = [c for c in short_df.columns if c != target][:12] + [target]
    corr = short_df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
                annot=len(cols)<=10, fmt='.1f', annot_kws={'size':6},
                linewidths=0.3, cbar_kws={'shrink':0.8})
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/05_correlation_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 05 saved: Correlation Heatmaps")


# ============================================================
# 3.  MODEL TRAINING & EVALUATION
# ============================================================

def get_models():
    return {
        'SVM':                Pipeline([('sc', StandardScaler()),
                               ('m', SVC(probability=True, random_state=42))]),
        'Logistic Regression':Pipeline([('sc', StandardScaler()),
                               ('m', LogisticRegression(max_iter=1000, random_state=42))]),
        'Random Forest':      Pipeline([('sc', StandardScaler()),
                               ('m', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'XGBoost':            Pipeline([('sc', StandardScaler()),
                               ('m', XGBClassifier(n_estimators=100, random_state=42,
                                                    eval_metric='logloss', verbosity=0))]),
    }

all_results  = {}
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 65)
print("   TRAINING MODELS ON ALL 3 DATASETS")
print("=" * 65)

for ds_name, (df, target, pos_label) in datasets.items():
    print(f"\n🔬 {ds_name}")
    print("   " + "-"*40)
    feat_cols = [c for c in df.columns if c != target]
    X = df[feat_cols]
    y = df[target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    ds_res = {}
    for m_name, pipeline in get_models().items():
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]
        cv_sc  = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='f1').mean()

        ds_res[m_name] = {
            'pipeline': pipeline,
            'y_pred':   y_pred,
            'y_prob':   y_prob,
            'accuracy': accuracy_score(y_te, y_pred),
            'precision':precision_score(y_te, y_pred, zero_division=0),
            'recall':   recall_score(y_te, y_pred, zero_division=0),
            'f1':       f1_score(y_te, y_pred, zero_division=0),
            'roc_auc':  roc_auc_score(y_te, y_prob),
            'cv_f1':    cv_sc,
            'y_test':   y_te,
            'feat_cols':feat_cols,
        }
        print(f"   ✅ {m_name:<22} Acc={ds_res[m_name]['accuracy']:.3f}  "
              f"F1={ds_res[m_name]['f1']:.3f}  AUC={ds_res[m_name]['roc_auc']:.3f}")

    all_results[ds_name] = ds_res

    # Save per-dataset results CSV
    pd.DataFrame([{
        'Model': mn,
        'Accuracy':  v['accuracy'],
        'Precision': v['precision'],
        'Recall':    v['recall'],
        'F1-Score':  v['f1'],
        'ROC-AUC':   v['roc_auc'],
        'CV-F1':     v['cv_f1'],
    } for mn, v in ds_res.items()]).round(4).to_csv(
        f"outputs/results/{ds_name.lower().replace(' ','_')}_results.csv", index=False)


# ============================================================
# 4.  RESULT PLOTS
# ============================================================

metrics     = ['accuracy','precision','recall','f1','roc_auc']
met_labels  = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
model_names = list(next(iter(all_results.values())).keys())

# ── Plots 6-8: Per-Dataset Model Comparison ──────────────────
for i, (ds_name, ds_res) in enumerate(all_results.items(), start=6):
    fig, ax = plt.subplots(figsize=(14, 6))
    x    = np.arange(len(metrics))
    w    = 0.18
    for j, (m_name, color) in enumerate(zip(model_names, COLORS)):
        vals = [ds_res[m_name][m] for m in metrics]
        bars = ax.bar(x + j*w, vals, w, label=m_name, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(met_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(f'{ds_name} — Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fname = f'outputs/plots/0{i}_{ds_name.lower().replace(" ","_")}_model_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot 0{i} saved: {ds_name} Model Comparison")

# ── Plot 9: ROC Curves (all datasets, all models) ────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ROC Curves — All Datasets & Models', fontsize=14, fontweight='bold')
for ax, (ds_name, ds_res) in zip(axes, all_results.items()):
    for m_name, color in zip(model_names, COLORS):
        v   = ds_res[m_name]
        fpr, tpr, _ = roc_curve(v['y_test'], v['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{m_name} ({v['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_title(ds_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('outputs/plots/09_roc_curves_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 09 saved: ROC Curves (all)")

# ── Plot 10: Confusion Matrices (best model per dataset) ─────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Confusion Matrices — All Models × All Datasets', fontsize=14, fontweight='bold')
for row, (ds_name, ds_res) in enumerate(all_results.items()):
    for col, m_name in enumerate(model_names):
        v  = ds_res[m_name]
        cm = confusion_matrix(v['y_test'], v['y_pred'])
        ConfusionMatrixDisplay(cm).plot(ax=axes[row][col], colorbar=False, cmap='Blues')
        axes[row][col].set_title(f'{ds_name}\n{m_name}', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 10 saved: Confusion Matrices")

# ── Plot 11: Feature Importance (RF & XGBoost per dataset) ───
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Feature Importance — Random Forest & XGBoost', fontsize=14, fontweight='bold')
for col, (ds_name, ds_res) in enumerate(all_results.items()):
    feat_cols = ds_res['Random Forest']['feat_cols']
    for row, m_name in enumerate(['Random Forest', 'XGBoost']):
        model  = ds_res[m_name]['pipeline'].named_steps['m']
        imps   = model.feature_importances_
        top_n  = min(10, len(feat_cols))
        idx    = np.argsort(imps)[-top_n:]
        ax     = axes[row][col]
        ax.barh([feat_cols[i].replace('_',' ').title() for i in idx],
                imps[idx], color=COLORS[1] if row==0 else COLORS[3], alpha=0.85)
        ax.set_title(f'{ds_name}\n{m_name}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('outputs/plots/11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 11 saved: Feature Importance")

# ── Plot 12: Overall Summary Heatmap ─────────────────────────
summary_rows = []
for ds_name, ds_res in all_results.items():
    for m_name, v in ds_res.items():
        summary_rows.append({
            'Dataset': ds_name,
            'Model':   m_name,
            'ROC-AUC': v['roc_auc'],
            'F1-Score':v['f1'],
            'Accuracy':v['accuracy'],
        })
summary_df = pd.DataFrame(summary_rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Overall Performance Summary — All Datasets & Models', fontsize=14, fontweight='bold')
for ax, metric in zip(axes, ['Accuracy','F1-Score','ROC-AUC']):
    pivot = summary_df.pivot(index='Model', columns='Dataset', values=metric)
    sns.heatmap(pivot, ax=ax, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.75, vmax=1.0, linewidths=0.5, cbar_kws={'shrink':0.8})
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
plt.tight_layout()
plt.savefig('outputs/plots/12_overall_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Plot 12 saved: Overall Summary Heatmap")


# ============================================================
# 5.  FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("   FINAL RESULTS SUMMARY")
print("=" * 65)

for ds_name, ds_res in all_results.items():
    print(f"\n🏥 {ds_name}")
    print(f"   {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
    print("   " + "-"*67)
    best = max(ds_res, key=lambda x: ds_res[x]['roc_auc'])
    for m_name, v in ds_res.items():
        star = ' 🏆' if m_name == best else ''
        print(f"   {m_name:<22} {v['accuracy']:>9.4f} {v['precision']:>10.4f} "
              f"{v['recall']:>8.4f} {v['f1']:>7.4f} {v['roc_auc']:>9.4f}{star}")

print("\n✅ All results saved to outputs/results/")
print("✅ All plots  saved to outputs/plots/")
print("✅ All datasets saved to outputs/datasets/")
print("=" * 65)
