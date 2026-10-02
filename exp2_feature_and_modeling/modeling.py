"""
================================================================================
EKSPERIMEN 2: FEATURE ENGINEERING & MULTI-MODEL CLASSIFICATION
Mata Kuliah : Penambangan Data (Data Mining) - Semester 5
Dataset     : UCI Heart Disease Cleaned
Penulis     : Muhammad Zakaria
Deskripsi   : Skrip untuk pemisahan data, encoding & scaling fitur, serta
              melatih dan membandingkan 5 algoritma klasifikasi data mining.
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Konfigurasi plotting
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATASET_PATH = BASE_DIR / "dataset" / "heart_disease_cleaned.csv"
RAW_DATASET_PATH = BASE_DIR / "dataset" / "heart_disease.csv"
OUTPUT_PLOT_DIR = Path(__file__).resolve().parent / "output_plots"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV_PATH = Path(__file__).resolve().parent / "baseline_results.csv"


def cetak_header(judul: str):
    print("\n" + "=" * 70)
    print(f" {judul.upper()} ")
    print("=" * 70)


def muat_data_bersih() -> pd.DataFrame:
    """Memuat data yang telah dibersihkan pada Eksperimen 1."""
    cetak_header("1. Memuat Dataset Hasil Pembersihan (Exp 1)")
    if CLEANED_DATASET_PATH.exists():
        df = pd.read_csv(CLEANED_DATASET_PATH)
        print(f"[INFO] Memuat dataset bersih dari: {CLEANED_DATASET_PATH.name}")
    elif RAW_DATASET_PATH.exists():
        print(
            "[WARN] File bersih tidak ditemukan, memuat data mentah dan auto-clean..."
        )
        df = pd.read_csv(RAW_DATASET_PATH)
        df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
        df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
        df["target"] = (df["target"] > 0).astype(int)
    else:
        raise FileNotFoundError("Dataset tidak ditemukan di folder dataset/!")

    print(f"[INFO] Data siap: {df.shape[0]} baris, {df.shape[1]} kolom.")
    return df


def rekayasa_fitur(df: pd.DataFrame):
    """
    Pra-pemrosesan data:
    - One-hot encoding untuk fitur kategorikal multi-kelas (cp, restecg, slope, thal).
    - Pemisahan Fitur (X) dan Label Target (y).
    - Train-Test Split (80:20, stratified).
    - Feature Scaling menggunakan StandardScaler pada fitur kontinu numerik.
    """
    cetak_header("2. Rekayasa Fitur & Pembagian Data (Train-Test Split)")

    # One-Hot Encoding fitur kategorikal non-biner
    kategori_fitur = ["cp", "restecg", "slope", "thal"]
    print(f"[ENCODING] Menerapkan One-Hot Encoding pada fitur: {kategori_fitur}")
    df_encoded = pd.get_dummies(df, columns=kategori_fitur, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]

    print(f"[INFO] Dimensi fitur X setelah encoding: {X.shape}")
    print(f"[INFO] Daftar kolom fitur:\n{list(X.columns)}")

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Data Latih (Train) : {X_train.shape[0]} sampel")
    print(f"[SPLIT] Data Uji (Test)    : {X_test.shape[0]} sampel")

    # Scaling Fitur Numerik Kontinu
    fitur_kontinu = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[fitur_kontinu] = scaler.fit_transform(X_train[fitur_kontinu])
    X_test_scaled[fitur_kontinu] = scaler.transform(X_test[fitur_kontinu])

    print(
        f"[SCALING] Fitur numerik {fitur_kontinu} berhasil dinormalisasi dengan StandardScaler."
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def latih_dan_evaluasi_model(X_train, X_test, y_train, y_test):
    """
    Melatih 5 algoritma data mining dan mengevaluasi performa dasarnya:
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Decision Tree
    4. Random Forest
    5. Support Vector Machine (SVM)
    """
    cetak_header("3. Pelatihan Model & Evaluasi Baseline")

    daftar_model = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", random_state=42),
    }

    hasil = []
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nama_model, model in daftar_model.items():
        print(f"[TRAINING] Melatih model: {nama_model}...")
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        # Hitung Metrik Evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # 5-Fold Cross Validation pada Training Set
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_strategy, scoring="accuracy"
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        hasil.append(
            {
                "Algoritma": nama_model,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "CV Mean Acc": cv_mean,
                "CV Std": cv_std,
            }
        )

    df_hasil = (
        pd.DataFrame(hasil)
        .sort_values(by="F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- TABEL PERBANDINGAN PERFORMA BASELINE MODEL ---")
    tabel_cetak = df_hasil.copy()
    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "CV Mean Acc",
    ]:
        tabel_cetak[col] = (tabel_cetak[col] * 100).round(2).astype(str) + "%"
    tabel_cetak["CV Std"] = tabel_cetak["CV Std"].round(4)
    print(tabel_cetak.to_string(index=False))

    # Simpan hasil ke CSV
    df_hasil.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n[SUKSES] Tabel hasil tersimpan di: {RESULTS_CSV_PATH.name}")
    return df_hasil


def visualisasikan_perbandingan(df_hasil: pd.DataFrame):
    """Membuat visualisasi perbandingan metrik performa model."""
    cetak_header("4. Visualisasi Komparasi Model")

    df_plot = df_hasil.melt(
        id_vars=["Algoritma"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metrik",
        value_name="Skor",
    )

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        x="Algoritma", y="Skor", hue="Metrik", data=df_plot, palette="viridis"
    )
    plt.title(
        "Perbandingan Metrik Evaluasi 5 Algoritma Klasifikasi Data Mining",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Model Klasifikasi", fontsize=11)
    plt.ylabel("Skor (0.0 - 1.0)", fontsize=11)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower right")
    plt.xticks(rotation=15)

    # Label nilai di atas bar
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                xytext=(0, 2),
                textcoords="offset points",
            )

    plot_file = OUTPUT_PLOT_DIR / "model_comparison.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[PLOT] Grafik komparasi berhasil disimpan di: {plot_file.name}")


def main():
    print("""
    +-------------------------------------------------------------+
    |       DATA MINING EXPERIMENT #2: MODELING & COMPARISON      |
    |      Implementasi 5 Algoritma Klasifikasi Data Mining       |
    +-------------------------------------------------------------+
    """)
    df = muat_data_bersih()
    X_train, X_test, y_train, y_test, _ = rekayasa_fitur(df)
    df_hasil = latih_dan_evaluasi_model(X_train, X_test, y_train, y_test)
    visualisasikan_perbandingan(df_hasil)

    model_terbaik = df_hasil.iloc[0]
    cetak_header("RINGKASAN KESIMPULAN EKSPERIMEN 2")
    print(f"""
    [1] Model dengan performa tertinggi pada pengujian baseline adalah:
        --> '{model_terbaik["Algoritma"]}'
        - Accuracy  : {model_terbaik["Accuracy"] * 100:.2f}%
        - Precision : {model_terbaik["Precision"] * 100:.2f}%
        - Recall    : {model_terbaik["Recall"] * 100:.2f}%
        - F1-Score  : {model_terbaik["F1-Score"] * 100:.2f}%
        - ROC-AUC   : {model_terbaik["ROC-AUC"] * 100:.2f}%
        - CV Acc    : {model_terbaik["CV Mean Acc"] * 100:.2f}% (+/- {model_terbaik["CV Std"] * 100:.2f}%)
    [2] Ensemble Learning (Random Forest) dan Kernel-based SVM memperlihatkan generalisasi yang
        sangat solid dibandingkan single Decision Tree yang rentan terhadap overfitting.
    [3] Model terbaik ini akan dilanjutkan ke Eksperimen 3 untuk analisis mendalam (Confusion Matrix,
        ROC Curve, Feature Importance) serta Hyperparameter Tuning menggunakan GridSearchCV.
    """)
    print("[SELESAI] Eksperimen 2 selesai dijalankan dengan sukses!\n")


if __name__ == "__main__":
    main()
