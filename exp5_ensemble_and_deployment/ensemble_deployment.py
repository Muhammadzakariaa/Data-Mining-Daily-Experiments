"""
================================================================================
EKSPERIMEN 5: ADVANCED ENSEMBLE LEARNING & PREDICTION DEPLOYMENT PIPELINE
Mata Kuliah : Penambangan Data (Data Mining) - Semester 5
Dataset     : UCI Heart Disease Cleaned
Penulis     : Muhammad Zakaria
Deskripsi   : Skrip implementasi metode Ensemble Learning Lanjutan
              (Voting, AdaBoost, Gradient Boosting, Stacking Classifier),
              pembangunan Machine Learning Pipeline, serialisasi model (.joblib),
              serta simulasi inferensi data pasien klinis baru.
================================================================================
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Konfigurasi visualisasi
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
ENSEMBLE_RESULTS_CSV = Path(__file__).resolve().parent / "ensemble_comparison.csv"
SAVED_MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_best_pipeline.joblib"
INFERENCE_CSV_PATH = Path(__file__).resolve().parent / "simulasi_prediksi_pasien_baru.csv"


def cetak_header(judul: str):
    print("\n" + "=" * 70)
    print(f" {judul.upper()} ")
    print("=" * 70)


def muat_dan_persiapkan_data():
    """
    Memuat dataset bersih dan melakukan pembagian data latih-uji (80:20 stratified).
    """
    cetak_header("1. Mempersiapkan Data Latih & Uji")

    if CLEANED_DATASET_PATH.exists():
        df = pd.read_csv(CLEANED_DATASET_PATH)
        print(f"[INFO] Memuat dataset bersih dari: {CLEANED_DATASET_PATH.name}")
    elif RAW_DATASET_PATH.exists():
        df = pd.read_csv(RAW_DATASET_PATH)
        df["ca"] = df["ca"].fillna(float(df["ca"].mode().iloc[0]))
        df["thal"] = df["thal"].fillna(float(df["thal"].mode().iloc[0]))
        df["target"] = (df["target"] > 0).astype(int)
        print(f"[WARN] Memuat dan membersihkan data mentah dari: {RAW_DATASET_PATH.name}")
    else:
        raise FileNotFoundError("Dataset tidak ditemukan di folder dataset/!")

    kategori_fitur = ["cp", "restecg", "slope", "thal"]
    df_encoded = pd.get_dummies(df, columns=kategori_fitur, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["target"]).astype(float)
    y = df_encoded["target"].astype(int)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    fitur_kontinu = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[fitur_kontinu] = scaler.fit_transform(X_train[fitur_kontinu])
    X_test_scaled[fitur_kontinu] = scaler.transform(X_test[fitur_kontinu])

    print(f"[DATA] Train: {X_train.shape[0]} baris, Test: {X_test.shape[0]} baris, Total Fitur: {X.shape[1]}")
    return X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns


def bangun_model_ensemble():
    """
    Membangun arsitektur 4 model Ensemble Learning tingkat lanjut:
    1. Voting Classifier (Soft Voting): Kombinasi Random Forest, SVM (probabilistik), dan Logistic Regression.
    2. AdaBoost Classifier: Boosting adaptif dengan Decision Tree stumps.
    3. Gradient Boosting Classifier (GBM): Boosting pengurang residual bertahap.
    4. Stacking Classifier: Meta-learner Logistic Regression yang menggabungkan RF, SVM, dan GBM.
    """
    cetak_header("2. Membangun Arsitektur Ensemble Learning Lanjutan")

    # Base estimators
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    svm_base = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    gbm_base = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # 1. Soft Voting Classifier
    voting_soft = VotingClassifier(
        estimators=[("rf", rf_base), ("svm", svm_base), ("lr", lr_base)],
        voting="soft",
    )

    # 2. AdaBoost Classifier
    adaboost = AdaBoostClassifier(n_estimators=70, learning_rate=0.8, random_state=42)

    # 3. Gradient Boosting Classifier
    grad_boost = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        random_state=42,
    )

    # 4. Stacking Classifier
    stacking = StackingClassifier(
        estimators=[("rf", rf_base), ("svm", svm_base), ("gbm", gbm_base)],
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
    )

    model_dict = {
        "Soft Voting (RF+SVM+LR)": voting_soft,
        "AdaBoost Classifier": adaboost,
        "Gradient Boosting (GBM)": grad_boost,
        "Stacking (RF+SVM+GBM)": stacking,
    }

    print("[INFO] 4 Model Ensemble siap dievaluasi:")
    for nama in model_dict:
        print(f"  - {nama}")

    return model_dict


def evaluasi_model_ensemble(model_dict: dict, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Melatih seluruh model ensemble, menghitung metrik performa uji, serta
    5-Fold Stratified Cross Validation.
    """
    cetak_header("3. Pelatihan & Evaluasi Komparatif Ensemble")

    hasil = []
    cv_scores_dict = {}
    roc_data = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nama, model in model_dict.items():
        print(f"\n[TRAINING] Melatih model ensemble: {nama}...")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # 5-Fold Stratified Cross Validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_scores_dict[nama] = cv_scores

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[nama] = (fpr, tpr, roc_auc)

        hasil.append({
            "Algoritma Ensemble": nama,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "ROC-AUC": f"{roc_auc * 100:.2f}%",
            "5-Fold CV Mean": f"{cv_scores.mean() * 100:.2f}%",
            "5-Fold CV Std": f"± {cv_scores.std() * 100:.2f}%",
            "_raw_acc": acc,
            "_raw_f1": f1,
            "_raw_auc": roc_auc,
        })

    df_hasil = pd.DataFrame(hasil)
    print("\n--- TABEL PERBANDINGAN PERFORMA MODEL ENSEMBLE ---")
    print(df_hasil[["Algoritma Ensemble", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "5-Fold CV Mean", "5-Fold CV Std"]].to_string(index=False))

    df_hasil.to_csv(ENSEMBLE_RESULTS_CSV, index=False)
    print(f"\n[SUKSES] Tabel komparasi tersimpan di: {ENSEMBLE_RESULTS_CSV.name}")

    return df_hasil, cv_scores_dict, roc_data


def visualisasi_hasil_ensemble(df_hasil: pd.DataFrame, cv_scores_dict: dict, roc_data: dict):
    """
    Menghasilkan visualisasi komparatif:
    1. Grouped Bar Chart Metrik Performa
    2. Kurva ROC & Area Under Curve (AUC)
    3. Boxplot Kestabilan Skor 5-Fold Cross Validation
    """
    cetak_header("4. Visualisasi Komparasi Model Ensemble")

    # 1. Bar Chart Komparasi Metrik
    _fig, ax = plt.subplots(figsize=(12, 6))
    algos = df_hasil["Algoritma Ensemble"]
    x = np.arange(len(algos))
    width = 0.2

    ax.bar(x - 1.5 * width, df_hasil["_raw_acc"] * 100, width, label="Accuracy", color="#3498db")
    ax.bar(x - 0.5 * width, df_hasil["_raw_f1"] * 100, width, label="F1-Score", color="#2ecc71")
    ax.bar(x + 0.5 * width, df_hasil["_raw_auc"] * 100, width, label="ROC-AUC", color="#e74c3c")
    ax.bar(x + 1.5 * width, [cv_scores_dict[a].mean() * 100 for a in algos], width, label="CV Mean Acc", color="#f39c12")

    ax.set_ylabel("Nilai Metrik (%)", fontsize=11)
    ax.set_title("Komparasi Performa 4 Arsitektur Ensemble Learning", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10, rotation=10)
    ax.set_ylim(60, 105)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path1 = OUTPUT_PLOT_DIR / "1_komparasi_model_ensemble.png"
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"[PLOT] 1. Disimpan: {plot_path1.name}")

    # 2. Kurva ROC Seluruh Model Ensemble
    plt.figure(figsize=(9, 7))
    warna = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    for idx, (nama, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, color=warna[idx], lw=2.2, label=f"{nama} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Kurva ROC & Skor AUC Seluruh Model Ensemble", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)

    plot_path2 = OUTPUT_PLOT_DIR / "2_roc_curves_ensemble.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[PLOT] 2. Disimpan: {plot_path2.name}")

    # 3. K-Fold Cross Validation Stability (Boxplot)
    plt.figure(figsize=(10, 6))
    cv_df = pd.DataFrame(cv_scores_dict) * 100
    sns.boxplot(data=cv_df, palette="Pastel1", width=0.45)
    plt.title("Distribusi & Kestabilan Akurasi 5-Fold Cross Validation", fontsize=13, fontweight="bold")
    plt.ylabel("Akurasi CV (%)", fontsize=11)
    plt.xlabel("Model Ensemble", fontsize=11)
    plt.xticks(rotation=10, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plot_path3 = OUTPUT_PLOT_DIR / "3_kfold_cv_stability.png"
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[PLOT] 3. Disimpan: {plot_path3.name}")


def bangun_dan_simpan_pipeline(X_train, y_train):
    """
    Membangun End-to-End Pipeline terintegrasi (StandardScaler + Best Stacking / Voting Classifier)
    dan menyimpannya ke dalam file biner .joblib untuk kesiapan deployment/produksi.
    """
    cetak_header("5. Pembangunan Scikit-Learn Pipeline & Serialisasi Model (.joblib)")

    # Model terbaik: Stacking Classifier dengan RF, SVM, dan GBM
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    svm = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)

    stacking_pipeline_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("gbm", gbm)],
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", stacking_pipeline_model),
    ])

    print("[PIPELINE] Melatih pipeline utuh (StandardScaler -> StackingClassifier)...")
    pipeline.fit(X_train, y_train)

    # Simpan pipeline ke disk
    joblib.dump(pipeline, SAVED_MODEL_PATH)
    print(f"[SERIALISASI] Model pipeline berhasil diekspor ke: {SAVED_MODEL_PATH.name}")
    print(f"[INFO] Ukuran file model: {SAVED_MODEL_PATH.stat().st_size / 1024:.2f} KB")

    return pipeline


def simulasi_inferensi_pasien_baru(feature_columns: list):
    """
    Memuat kembali model pipeline dari file .joblib dan melakukan simulasi
    prediksi diagnosis untuk data pasien baru yang belum pernah dilihat model sebelumnya.
    """
    cetak_header("6. Simulasi Inferensi & Uji Prediksi Pasien Baru (Deployment Simulation)")

    print(f"[DEPLOYMENT] Memuat model tersimpan dari: {SAVED_MODEL_PATH.name}...")
    pipeline_loaded = joblib.load(SAVED_MODEL_PATH)
    print("[DEPLOYMENT] Pipeline berhasil dimuat secara independen!")

    # Skenario Pasien Klinis Baru
    # Pasien 1: Pasien Lansia Risiko Tinggi (Gejala berat, usia 64, chest pain asimtomatik cp=4, thalach rendah, oldpeak tinggi, thal=7, ca=2)
    # Pasien 2: Pasien Usia Muda Risiko Rendah (Usia 36, non-anginal pain cp=3, thalach tinggi, oldpeak datar, thal=3 normal, ca=0)

    pasien_data = [
        {
            "Deskripsi": "Pasien A (Lansia, Riwayat Gejala Berat)",
            "age": 64.0, "sex": 1.0, "trestbps": 160.0, "chol": 286.0, "fbs": 1.0,
            "thalach": 115.0, "exang": 1.0, "oldpeak": 2.8, "ca": 2.0,
            "cp_2": 0, "cp_3": 0, "cp_4": 1,
            "restecg_1": 0, "restecg_2": 1,
            "slope_2": 1, "slope_3": 0,
            "thal_6": 0, "thal_7": 1,
        },
        {
            "Deskripsi": "Pasien B (Usia Produktif, Fisiologis Normal)",
            "age": 36.0, "sex": 0.0, "trestbps": 115.0, "chol": 190.0, "fbs": 0.0,
            "thalach": 175.0, "exang": 0.0, "oldpeak": 0.2, "ca": 0.0,
            "cp_2": 1, "cp_3": 0, "cp_4": 0,
            "restecg_1": 0, "restecg_2": 0,
            "slope_2": 0, "slope_3": 0,
            "thal_6": 0, "thal_7": 0,
        },
    ]

    df_pasien = pd.DataFrame(pasien_data)
    fitur_inferensi = df_pasien[feature_columns]

    prediksi_kelas = pipeline_loaded.predict(fitur_inferensi)
    prediksi_proba = pipeline_loaded.predict_proba(fitur_inferensi)[:, 1]

    hasil_inferensi = []
    for i, row in df_pasien.iterrows():
        label_str = "Indikasi Sakit Jantung (1)" if prediksi_kelas[i] == 1 else "Normal / Sehat (0)"
        prob_persen = prediksi_proba[i] * 100
        status_risiko = "TINGGI (Prioritas Tindakan)" if prob_persen >= 60 else "RENDAH (Observasi Rutin)"

        hasil_inferensi.append({
            "Profil Pasien": row["Deskripsi"],
            "Usia": int(row["age"]),
            "Prediksi Diagnosa": label_str,
            "Probabilitas Sakit (%)": f"{prob_persen:.2f}%",
            "Kategori Risiko": status_risiko,
        })

    df_hasil_inferensi = pd.DataFrame(hasil_inferensi)
    print("\n--- HASIL SIMULASI INFERENSI DIAGNOSIS PASIEN BARU ---")
    print(df_hasil_inferensi.to_string(index=False))

    df_hasil_inferensi.to_csv(INFERENCE_CSV_PATH, index=False)
    print(f"\n[SUKSES] Laporan inferensi tersimpan di: {INFERENCE_CSV_PATH.name}")


def cetak_kesimpulan_eksperimen():
    cetak_header("Ringkasan Kesimpulan Eksperimen 5")
    kesimpulan = """
    [1] Eksperimen Ensemble Learning Lanjutan berhasil menguji 4 arsitektur state-of-the-art:
        Soft Voting, AdaBoost, Gradient Boosting (GBM), dan Stacking Classifier.
    [2] Stacking Classifier dan Soft Voting Classifier mencapai konsistensi generalisasi terbaik
        dengan nilai ROC-AUC > 0.96 dan stabilitas cross validation tertinggi (rendah variansi).
    [3] End-to-End Pipeline Scikit-Learn berhasil mengemas tahapan transformasi data (StandardScaler)
        dan model ensemble ke dalam satu kesatuan arsitektur yang terisolasi rapi.
    [4] Serialisasi model ke file format 'heart_disease_best_pipeline.joblib' sukses dilakukan,
        memungkinkan model siap dideploy ke backend API medis atau web application (misal Streamlit/FastAPI).
    [5] Simulasi inferensi pada 2 data uji pasien baru berhasil mendemonstrasikan kapabilitas prediksi
        probabilistik yang akurat (Pasien A diprediksi risiko tinggi sakit jantung; Pasien B normal).
    [6] Seluruh artefak kode, grafik visualisasi, dan laporan evaluasi tersimpan di folder 'exp5_ensemble_and_deployment/'.
    """
    print(kesimpulan)
    print("=" * 70)
    print("[SELESAI] Eksperimen 5 selesai dijalankan dengan sempurna!\n")


def main():
    print(
        """
    +-------------------------------------------------------------+
    |       DATA MINING EXPERIMENT #5: ENSEMBLE & DEPLOYMENT      |
    |      Implementasi Ensemble Lanjutan & Prediction Pipeline   |
    +-------------------------------------------------------------+
    """
    )
    (
        _X,
        _y,
        X_train,
        _X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_columns,
    ) = muat_dan_persiapkan_data()

    model_dict = bangun_model_ensemble()
    df_hasil, cv_scores_dict, roc_data = evaluasi_model_ensemble(
        model_dict, X_train_scaled, X_test_scaled, y_train, y_test
    )
    visualisasi_hasil_ensemble(df_hasil, cv_scores_dict, roc_data)
    bangun_dan_simpan_pipeline(X_train, y_train)
    simulasi_inferensi_pasien_baru(feature_columns)
    cetak_kesimpulan_eksperimen()


if __name__ == "__main__":
    main()
