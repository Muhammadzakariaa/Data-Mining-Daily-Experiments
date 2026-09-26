"""
================================================================================
EKSPERIMEN 1: EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING
Mata Kuliah : Penambangan Data (Data Mining) - Semester 5
Dataset     : UCI Heart Disease Dataset (Cleveland)
Penulis     : Muhammad Zakaria
Deskripsi   : Skrip untuk inspeksi data, penanganan missing values & outlier,
              analisis korelasi, serta visualisasi eksplorasi data.
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Atur gaya visualisasi
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
sns.set_palette("Set2")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

# Direktori kerja berbasis path relatif file
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "heart_disease.csv"
CLEANED_DATASET_PATH = BASE_DIR / "dataset" / "heart_disease_cleaned.csv"
OUTPUT_PLOT_DIR = Path(__file__).resolve().parent / "output_plots"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def cetak_header(judul: str):
    print("\n" + "=" * 70)
    print(f" {judul.upper()} ")
    print("=" * 70)


def muat_data() -> pd.DataFrame:
    """Memuat dataset mentah dari folder dataset."""
    cetak_header("1. Memuat Dataset Jantung (Heart Disease)")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print("[INFO] Dataset berhasil dimuat!")
    print(f"[INFO] Dimensi data : {df.shape[0]} baris, {df.shape[1]} kolom\n")
    print("--- 5 Baris Pertama Data Mentah ---")
    print(df.head())
    print("\n--- Informasi Tipe Data & Memori ---")
    df.info()
    return df


def inspeksi_kualitas_data(df: pd.DataFrame):
    """Pemeriksaan missing values dan data duplikat."""
    cetak_header("2. Pemeriksaan Kualitas Data")

    # Cek Missing Value
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": missing, "Persentase (%)": missing_pct.round(2)}
    )
    print("--- Ringkasan Missing Values ---")
    print(missing_df[missing_df["Missing Values"] > 0])

    # Cek Duplikasi
    duplikat = df.duplicated().sum()
    print(f"\n[INFO] Jumlah baris duplikat: {duplikat}")
    return missing_df


def bersihkan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan data cleaning:
    - Imputasi missing values pada kolom diskret 'ca' dan 'thal' menggunakan modus.
    - Binarisasi target (0 = Tidak Berpenyakit Jantung, 1..4 = Berpenyakit Jantung -> 1).
    - Menyesuaikan tipe data agar efisien dan konsisten.
    """
    cetak_header("3. Pembersihan Data & Imputasi")
    df_clean = df.copy()

    # Imputasi Missing Values
    for col in ["ca", "thal"]:
        if df_clean[col].isnull().sum() > 0:
            modus_val = df_clean[col].mode()[0]
            print(
                f"[IMPUTASI] Kolom '{col}' memiliki missing value. Diimputasi dengan Modus = {modus_val}"
            )
            df_clean[col] = df_clean[col].fillna(modus_val)

    # Binarisasi Target (Cleveland dataset memiliki target 0 = Sehat, 1-4 = Berpenyakit)
    # Standar benchmark biner: 0 = Normal/Negatif, 1 = Positif Penyakit Jantung
    distribusi_awal = df_clean["target"].value_counts().to_dict()
    print(f"[TARGET] Distribusi awal nilai target: {distribusi_awal}")

    df_clean["target"] = (df_clean["target"] > 0).astype(int)
    distribusi_biner = df_clean["target"].value_counts().to_dict()
    print(
        f"[TARGET] Distribusi target setelah binarisasi (0: Normal, 1: Sakit): {distribusi_biner}"
    )

    # Casting tipe data integer yang sesuai
    int_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    for col in int_cols:
        df_clean[col] = df_clean[col].astype(int)

    # Simpan dataset hasil pembersihan
    df_clean.to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"[SUKSES] Dataset bersih tersimpan di: {CLEANED_DATASET_PATH}")
    return df_clean


def deteksi_outlier(df: pd.DataFrame):
    """Menganalisis outlier pada fitur-fitur numerik kontinu menggunakan metode IQR."""
    cetak_header("4. Analisis Outlier (Metode Interquartile Range - IQR)")
    kolom_numerik = ["trestbps", "chol", "thalach", "oldpeak"]

    for col in kolom_numerik:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(
            f"Fitur '{col:10s}': IQR={iqr:.2f} | Batas [{lower_bound:.2f}, {upper_bound:.2f}] | Outlier: {len(outliers)} baris ({len(outliers) / len(df) * 100:.1f}%)"
        )


def visualisasi_eda(df: pd.DataFrame):
    """Membuat visualisasi komprehensif untuk laporan praktikum Data Mining."""
    cetak_header("5. Visualisasi Exploratory Data Analysis (EDA)")

    # 1. Distribusi Target
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="target", hue="target", data=df, palette=["#2ecc71", "#e74c3c"], legend=False
    )
    plt.title(
        "Distribusi Label Diagnosa Penyakit Jantung", fontsize=13, fontweight="bold"
    )
    plt.xlabel("Diagnosa (0 = Normal, 1 = Penyakit Jantung)", fontsize=11)
    plt.ylabel("Jumlah Pasien", fontsize=11)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=11,
            color="black",
            xytext=(0, 4),
            textcoords="offset points",
            fontweight="bold",
        )
    plot_path1 = OUTPUT_PLOT_DIR / "1_distribusi_target.png"
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"[PLOT] 1. Disimpan: {plot_path1.name}")

    # 2. Distribusi Fitur Numerik Penting
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fitur_kontinu = [
        ("age", "Distribusi Usia (Age)"),
        ("trestbps", "Tekanan Darah Istirahat (Trestbps)"),
        ("chol", "Kolesterol Serum (Chol)"),
        ("thalach", "Detak Jantung Maksimum (Thalach)"),
    ]

    for idx, (col, judul) in enumerate(fitur_kontinu):
        r, c = idx // 2, idx % 2
        sns.histplot(
            df[col], kde=True, ax=axes[r, c], color="#3498db", edgecolor="black"
        )
        axes[r, c].set_title(judul, fontsize=11, fontweight="bold")
        axes[r, c].set_xlabel(col)
        axes[r, c].set_ylabel("Frekuensi")
    fig.tight_layout()
    plot_path2 = OUTPUT_PLOT_DIR / "2_distribusi_fitur_numerik.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[PLOT] 2. Disimpan: {plot_path2.name}")

    # 3. Hubungan Usia dan Detak Jantung Maksimum berdasarkan Status Penyakit Jantung
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="age",
        y="thalach",
        hue="target",
        data=df,
        palette=["#27ae60", "#c0392b"],
        alpha=0.85,
        s=70,
        style="target",
    )
    plt.title(
        "Korelasi Usia vs Detak Jantung Maksimum (Thalach) terhadap Diagnosa",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Usia (Tahun)", fontsize=11)
    plt.ylabel("Detak Jantung Maksimum (bpm)", fontsize=11)
    plt.legend(title="Diagnosa", labels=["Normal (0)", "Penyakit Jantung (1)"])
    plot_path3 = OUTPUT_PLOT_DIR / "3_usia_vs_thalach.png"
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[PLOT] 3. Disimpan: {plot_path3.name}")

    # 4. Correlation Heatmap Matrix
    plt.figure(figsize=(12, 9))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Matriks Korelasi Antar-Fitur Dataset Heart Disease",
        fontsize=14,
        fontweight="bold",
    )
    plot_path4 = OUTPUT_PLOT_DIR / "4_correlation_heatmap.png"
    plt.savefig(plot_path4, dpi=300)
    plt.close()
    print(f"[PLOT] 4. Disimpan: {plot_path4.name}")


def main():
    print("""
    +-------------------------------------------------------------+
    |         DATA MINING EXPERIMENT #1: EDA & CLEANING           |
    |         Praktikum Mandiri Analisis & Eksplorasi Data        |
    +-------------------------------------------------------------+
    """)
    df_raw = muat_data()
    inspeksi_kualitas_data(df_raw)
    df_cleaned = bersihkan_data(df_raw)
    deteksi_outlier(df_cleaned)
    visualisasi_eda(df_cleaned)

    cetak_header("RINGKASAN KESIMPULAN EKSPERIMEN 1")
    print("""
    [1] Dataset awal memiliki 303 sampel dengan 6 missing values (ca=4, thal=2) yang telah diimputasi modus.
    [2] Variabel target berhasil dibinerisasi menjadi 0 (Normal: 164 sampel) dan 1 (Penyakit: 139 sampel).
    [3] Pasien dengan penyakit jantung cenderung memiliki nilai Thalach (detak jantung maksimum) yang lebih rendah
        dan tingkat ST depression (oldpeak) yang lebih tinggi pada saat latihan fisik.
    [4] Korelasi tertinggi terhadap target terdapat pada variabel:
        - cp (chest pain type)
        - thalach (maximum heart rate)
        - oldpeak (ST depression)
        - ca (number of major vessels colored by flourosopy)
    [5] Dataset bersih telah disimpan di 'dataset/heart_disease_cleaned.csv' untuk tahap modeling.
    """)
    print("[SELESAI] Eksperimen 1 selesai dijalankan dengan sukses!\n")


if __name__ == "__main__":
    main()
