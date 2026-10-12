# 🩺 Data Mining Daily Experiments: Heart Disease Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Praktikum-Semester%205-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Repositori ini berisi serangkaian eksperimen berkala (*daily experiments*) untuk tugas praktikum mata kuliah **Penambangan Data (Data Mining)** - Semester 5, Program Studi Teknik Informatika.

Studi kasus yang diangkat adalah **Klasifikasi Prediksi Penyakit Jantung** menggunakan dataset standar **UCI Cleveland Heart Disease**, dengan mengimplementasikan tahapan *Knowledge Discovery in Databases (KDD)* secara modular dan bertahap.

---

## 📂 Struktur Direktori Proyek

```text
Data-Mining-Daily-Experiments/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── heart_disease.csv
│   └── heart_disease_cleaned.csv
├── exp1_eda_and_cleaning/
│   ├── cleaning_eda.py
│   └── output_plots/
│       ├── 1_distribusi_target.png
│       ├── 2_distribusi_fitur_numerik.png
│       ├── 3_usia_vs_thalach.png
│       └── 4_correlation_heatmap.png
├── exp2_feature_and_modeling/
│   ├── modeling.py
│   ├── baseline_results.csv
│   └── output_plots/
│       └── model_comparison.png
└── exp3_evaluation_and_tuning/
    ├── evaluation.py
    ├── tuning_comparison.csv
    └── output_plots/
        ├── 1_confusion_matrix_rf_svm.png
        ├── 2_roc_curves_comparison.png
        ├── 3_feature_importance.png
        └── 4_tuning_performance_delta.png
```

---

## 📊 Kamus Data (Dataset Metadata)

Dataset yang digunakan bersumber dari **UCI Machine Learning Repository: Heart Disease (Cleveland Database)** yang terdiri dari 303 sampel pasien dan 14 variabel klinis:

| No | Atribut | Tipe Data | Deskripsi Klinis |
|---|---|---|---|
| 1 | `age` | Numerik | Usia pasien dalam satuan tahun |
| 2 | `sex` | Kategorikal | Jenis kelamin (1 = Laki-laki; 0 = Perempuan) |
| 3 | `cp` | Kategorikal | Tipe nyeri dada (1: Typical Angina, 2: Atypical Angina, 3: Non-anginal, 4: Asymptomatic) |
| 4 | `trestbps` | Numerik | Tekanan darah saat istirahat (*resting blood pressure*) dalam mmHg |
| 5 | `chol` | Numerik | Kadar kolesterol serum dalam mg/dl |
| 6 | `fbs` | Kategorikal | Gula darah puasa > 120 mg/dl (1 = Benar; 0 = Salah) |
| 7 | `restecg` | Kategorikal | Hasil elektrokardiografi istirahat (0: Normal, 1: ST-T wave abnormal, 2: Hipertrofi ventrikel kiri) |
| 8 | `thalach` | Numerik | Detak jantung maksimum yang dicapai (*maximum heart rate achieved*) |
| 9 | `exang` | Kategorikal | Nyeri dada yang diinduksi oleh latihan/olahraga (1 = Ya; 0 = Tidak) |
| 10 | `oldpeak` | Numerik | Depresi segmen ST yang diinduksi oleh latihan fisik relatif terhadap istirahat |
| 11 | `slope` | Kategorikal | Kemiringan puncak segmen ST saat latihan (1: Upsloping, 2: Flat, 3: Downsloping) |
| 12 | `ca` | Kategorikal | Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi |
| 13 | `thal` | Kategorikal | Defek thallium (3: Normal, 6: Fixed Defect, 7: Reversible Defect) |
| 14 | `target` | Kategorikal | Diagnosa penyakit jantung (0 = Normal/Sehat; 1 = Terindikasi Sakit Jantung) |

---

## 🔬 Rincian Eksperimen Harian

### 🧪 Eksperimen 1: EDA & Data Cleaning (`exp1_eda_and_cleaning/`)
- **Tujuan**: Memahami karakteristik distribusi data, mendeteksi anomali, dan membersihkan dataset.
- **Langkah-langkah**:
  1. Pemeriksaan *missing values*: Menemukan 6 data kosong (`ca` = 4, `thal` = 2) dan melakukan imputasi berbasis modus.
  2. Binarisasi label target dari skala multi-tahap (0-4) menjadi biner (0 = Normal, 1 = Berpenyakit).
  3. Analisis *outlier* menggunakan metode IQR (*Interquartile Range*) pada fitur kontinu (`trestbps`, `chol`, `thalach`, `oldpeak`).
  4. Visualisasi EDA: Distribusi target, distribusi fitur numerik dengan KDE, *scatterplot* usia vs *thalach*, dan *correlation heatmap*.
  5. Menyimpan dataset yang telah dibersihkan ke `dataset/heart_disease_cleaned.csv`.

### 🤖 Eksperimen 2: Feature Engineering & Multi-Model Classification (`exp2_feature_and_modeling/`)
- **Tujuan**: Mempersiapkan fitur dan membandingkan performa 5 algoritma klasifikasi *data mining*.
- **Langkah-langkah**:
  1. *One-Hot Encoding* untuk variabel kategorikal non-ordinal (`cp`, `restecg`, `slope`, `thal`).
  2. Pembagian data latih dan uji (*Stratified Train-Test Split* 80:20) dengan `random_state=42`.
  3. *Feature scaling* menggunakan `StandardScaler` pada fitur numerik kontinu.
  4. Pelatihan dan evaluasi komparatif 5 algoritma baseline:
     - **Logistic Regression** (Linear Probabilistik)
     - **K-Nearest Neighbors (KNN)** (Instance-based Learning)
     - **Decision Tree** (Tree-based Rule Extraction)
     - **Random Forest** (Ensemble Bagging)
     - **Support Vector Machine (SVM)** (Margin-based Kernel Classifier)
  5. Pengujian *5-Fold Stratified Cross-Validation* untuk menguji stabilitas generalisasi model.

### 🎯 Eksperimen 3: Evaluation, Hyperparameter Tuning & Interpretability (`exp3_evaluation_and_tuning/`)
- **Tujuan**: Menganalisis model terbaik secara mendalam, melakukan tuning parameter, dan mengidentifikasi fitur paling berpengaruh.
- **Langkah-langkah**:
  1. Evaluasi diagnostik menggunakan **Confusion Matrix** beranotasi dan **Classification Report** lengkap.
  2. Analisis performa ambang batas menggunakan **ROC Curve** dan skor **Area Under Curve (AUC)**.
  3. Optimasi hyperparameter menggunakan **GridSearchCV** pada algoritma Random Forest (kombinasi `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, dan `criterion`).
  4. Komparasi performa *Before vs After Hyperparameter Tuning*.
  5. Ekstraksi dan visualisasi **Feature Importance** (Gini Impurity) untuk interpretasi klinis.

---

## 📈 Ringkasan Hasil Eksperimen

### 1. Perbandingan Model Baseline (Eksperimen 2)
| Algoritma | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Acc (± Std) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM)** | **93.44%** | **90.00%** | **96.43%** | **93.10%** | **96.86%** | **81.82% (± 2.70%)** |
| **K-Nearest Neighbors (KNN)** | 91.80% | 87.10% | 96.43% | 91.53% | 96.65% | 78.53% (± 3.73%) |
| **Random Forest** | 86.89% | 83.33% | 89.29% | 86.21% | 95.24% | 80.14% (± 2.28%) |
| **Logistic Regression** | 85.25% | 80.65% | 89.29% | 84.75% | 95.67% | 84.29% (± 2.18%) |
| **Decision Tree** | 75.41% | 78.26% | 64.29% | 70.59% | 77.81% | 75.21% (± 4.12%) |

### 2. Peningkatan Random Forest Setelah Tuning (Eksperimen 3)
| Status | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline Random Forest | 86.89% | 83.33% | 89.29% | 86.21% | 95.24% |
| **Tuned Random Forest (GridSearchCV)** | **88.52%** | **86.21%** | **89.29%** | **87.72%** | **96.00%** |
| *Delta Peningkatan* | *+1.63%* | *+2.88%* | *0.00%* | *+1.51%* | *+0.76%* |

### 3. Top 5 Fitur Penentu Diagnosa Jantung (Feature Importance)
1. **`thal_7`** (Reversible Defect Thallium) - *14.57%*
2. **`thalach`** (Maximum Heart Rate Achieved) - *14.50%*
3. **`cp_4`** (Asymptomatic Chest Pain) - *11.92%*
4. **`ca`** (Major Vessels Colored by Fluoroscopy) - *10.50%*
5. **`oldpeak`** (ST Depression Induced by Exercise) - *9.64%*

---

## 🚀 Panduan Menjalankan Proyek

### 1. Masuk ke Direktori Proyek
```bash
cd Data-Mining-Daily-Experiments
```

### 2. Menyiapkan Virtual Environment & Dependensi
```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktivasi venv (Linux/macOS)
source venv/bin/activate

# Instalasi library
pip install -r requirements.txt
```

### 3. Eksekusi Eksperimen Secara Berurutan
```bash
# Eksperimen 1: EDA dan Pembersihan Data
python exp1_eda_and_cleaning/cleaning_eda.py

# Eksperimen 2: Rekayasa Fitur dan Pelatihan 5 Model Baseline
python exp2_feature_and_modeling/modeling.py

# Eksperimen 3: Evaluasi Diagnostik, Tuning Parameter, dan Analisis Fitur
python exp3_evaluation_and_tuning/evaluation.py
```
