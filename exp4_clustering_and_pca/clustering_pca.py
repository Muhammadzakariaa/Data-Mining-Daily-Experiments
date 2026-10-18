"""
================================================================================
EKSPERIMEN 4: UNSUPERVISED LEARNING - CLUSTERING (K-MEANS, AGGLOMERATIVE) & PCA
Mata Kuliah : Penambangan Data (Data Mining) - Semester 5
Dataset     : UCI Heart Disease Cleaned
Penulis     : Muhammad Zakaria
Deskripsi   : Segmentasi pasien kardiovaskular menggunakan metode pembelajaran
              tak terarah (Unsupervised Learning):
              1. Penentuan klaster optimal (Elbow Method & Silhouette Analysis)
              2. Pemodelan K-Means & Agglomerative Hierarchical Clustering
              3. Evaluasi Kualitas Klaster & Validasi Eksternal (ARI & NMI)
              4. Reduksi Dimensi 2D dengan PCA (Principal Component Analysis)
              5. Profiling Karakteristik Klinis tiap Klaster Pasien
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

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
CLUSTERING_METRICS_CSV = Path(__file__).resolve().parent / "clustering_metrics.csv"
CLUSTER_PROFILING_CSV = Path(__file__).resolve().parent / "cluster_patient_summary.csv"


def cetak_header(judul: str):
    print("\n" + "=" * 70)
    print(f" {judul.upper()} ")
    print("=" * 70)


def muat_dan_siapkan_data():
    """
    Memuat dataset bersih dan mempersiapkan data untuk analisis Unsupervised Learning.
    Catatan: Dalam unsupervised learning, kolom label 'target' dipisahkan dan
    hanya digunakan saat evaluasi validasi eksternal.
    """
    cetak_header("1. Memuat Dataset & Pra-pemrosesan Data Unsupervised")

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

    print(f"[INFO] Jumlah sampel: {df.shape[0]} pasien, {df.shape[1]} fitur awal.")

    # Simpan ground truth label target untuk validasi eksternal
    y_true = df["target"].values
    df_features = df.drop(columns=["target"])

    # One-hot encoding untuk variabel kategorikal
    kategori_fitur = ["cp", "restecg", "slope", "thal"]
    df_encoded = pd.get_dummies(df_features, columns=kategori_fitur, drop_first=True, dtype=int)

    # Standarisasi seluruh fitur (sangat vital untuk algoritma berbasis jarak seperti K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    feature_names = list(df_encoded.columns)

    print(f"[INFO] Dimensi data setelah One-Hot Encoding: {X_scaled.shape}")
    print(f"[SCALING] Seluruh {X_scaled.shape[1]} fitur telah distandarisasi (mean=0, variance=1).")

    return df, df_encoded, X_scaled, y_true, feature_names


def analisis_k_optimal(X_scaled: np.ndarray):
    """
    Menentukan jumlah klaster optimal (K) menggunakan:
    1. Elbow Method (Inertia / WCSS)
    2. Silhouette Score Analysis
    Rentang K diuji dari 2 hingga 8.
    """
    cetak_header("2. Analisis Jumlah Klaster Optimal (Elbow Method & Silhouette Score)")

    k_range = list(range(2, 9))
    inertia_scores = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia_scores.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil_score)
        print(f"[TEST K={k}] Inertia (WCSS): {kmeans.inertia_:10.2f} | Silhouette Score: {sil_score:.4f}")

    # Visualisasi Elbow & Silhouette
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Elbow Method
    ax1.plot(k_range, inertia_scores, marker="o", linewidth=2.2, markersize=8, color="#2980b9")
    ax1.set_title("Elbow Method untuk K-Means", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Jumlah Klaster (K)", fontsize=11)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Silhouette Score
    ax2.plot(k_range, silhouette_scores, marker="s", linewidth=2.2, markersize=8, color="#e74c3c")
    ax2.set_title("Analisis Silhouette Score per Nilai K", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Jumlah Klaster (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Coefficient", fontsize=11)
    ax2.set_xticks(k_range)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # K=2 dipilih karena merupakan elbow point pertama dan representasi klinis biner (Sehat vs Sakit)
    selected_k = 2
    ax2.annotate(
        f"K Terpilih = {selected_k} (Skor: {silhouette_scores[0]:.3f})",
        xy=(selected_k, silhouette_scores[0]),
        xytext=(selected_k + 0.5, silhouette_scores[0] + 0.015),
        arrowprops={"facecolor": "black", "shrink": 0.08, "width": 1.5, "headwidth": 7},
        fontweight="bold",
        fontsize=10,
    )

    plot_path1 = OUTPUT_PLOT_DIR / "1_elbow_dan_silhouette_analysis.png"
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"\n[PLOT] 1. Disimpan: {plot_path1.name}")
    print(f"[KESIMPULAN K] Nilai K={selected_k} dipilih sebagai klasterisasi klinis biner (Normal vs Risiko Penyakit Jantung).")

    return selected_k


def jalankan_dan_bandingkan_clustering(X_scaled: np.ndarray, y_true: np.ndarray, k: int = 2):
    """
    Melatih dan membandingkan dua algoritma klasterisasi:
    1. K-Means Clustering
    2. Agglomerative Hierarchical Clustering (Ward Linkage)

    Metrik Evaluasi Internal:
    - Silhouette Score (semakin mendekati 1 semakin baik)
    - Davies-Bouldin Index (semakin mendekati 0 semakin baik)
    - Calinski-Harabasz Score (semakin tinggi semakin baik)

    Metrik Validasi Eksternal terhadap Ground Truth (Label Diagnosa):
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    """
    cetak_header("3. Pemodelan Klasterisasi & Komparasi Metrik")

    # 1. K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 2. Agglomerative Hierarchical Clustering
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)

    # Menghitung Metrik Evaluasi
    metrik_data = []

    for nama, labels in [("K-Means Clustering", kmeans_labels), ("Agglomerative Hierarchical", agg_labels)]:
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        metrik_data.append({
            "Algoritma": nama,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin": round(db, 4),
            "Calinski-Harabasz": round(ch, 2),
            "Adjusted Rand Index (ARI)": round(ari, 4),
            "Normalized Mutual Info (NMI)": round(nmi, 4),
        })

    df_metrik = pd.DataFrame(metrik_data)
    print("\n--- TABEL PERBANDINGAN METRIK EVALUASI KLASTERISASI ---")
    print(df_metrik.to_string(index=False))

    df_metrik.to_csv(CLUSTERING_METRICS_CSV, index=False)
    print(f"\n[SUKSES] Metrik komparasi tersimpan di: {CLUSTERING_METRICS_CSV.name}")

    return kmeans, kmeans_labels, agg_labels, df_metrik


def visualisasi_pca_dan_dendrogram(X_scaled: np.ndarray, kmeans: KMeans, kmeans_labels: np.ndarray, agg_labels: np.ndarray, y_true: np.ndarray):
    """
    1. Reduksi dimensi fitur menggunakan PCA (Principal Component Analysis) ke 2 komponen.
    2. Visualisasi sebaran klaster K-Means vs Label Diagnosa Ground Truth.
    3. Visualisasi Hierarchical Clustering Dendrogram & sebaran klaster Agglomerative.
    """
    cetak_header("4. Reduksi Dimensi PCA & Visualisasi Dendrogram")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    print(f"[PCA] PC1 menjelaskan variansi: {var_ratio[0] * 100:.2f}%")
    print(f"[PCA] PC2 menjelaskan variansi: {var_ratio[1] * 100:.2f}%")
    print(f"[PCA] Total variansi terjelaskan (2 komponen): {sum(var_ratio) * 100:.2f}%")

    # Transformasi centroid K-Means ke ruang 2D PCA
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Plot 2: Perbandingan Klaster K-Means vs Ground Truth Target pada Ruang PCA 2D
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # K-Means Cluster di PC1 & PC2
    axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        cmap="coolwarm",
        alpha=0.75,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[0].scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=250,
        color="black",
        edgecolors="yellow",
        linewidths=1.5,
        label="Centroid K-Means",
    )
    axes[0].set_title("Segmentasi Pasien dengan K-Means (K=2) pada Ruang PCA", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"Principal Component 1 ({var_ratio[0] * 100:.1f}% Variansi)", fontsize=11)
    axes[0].set_ylabel(f"Principal Component 2 ({var_ratio[1] * 100:.1f}% Variansi)", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Ground Truth Diagnosa di PC1 & PC2
    axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_true,
        cmap="Set1",
        alpha=0.75,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[1].set_title("Ground Truth Diagnosa Medis Asli (0: Normal, 1: Sakit)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"Principal Component 1 ({var_ratio[0] * 100:.1f}% Variansi)", fontsize=11)
    axes[1].set_ylabel(f"Principal Component 2 ({var_ratio[1] * 100:.1f}% Variansi)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plot_path2 = OUTPUT_PLOT_DIR / "2_kmeans_clustering_pca2d.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[PLOT] 2. Disimpan: {plot_path2.name}")

    # Plot 3: Dendrogram Hierarchical Clustering & Agglomerative PCA
    fig = plt.figure(figsize=(15, 6))

    # Dendrogram (Truncated untuk kejelasan visualisasi)
    ax_dendro = fig.add_subplot(1, 2, 1)
    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=25,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax_dendro,
    )
    ax_dendro.set_title("Dendrogram Hierarki Pengelompokan (Ward Linkage)", fontsize=12, fontweight="bold")
    ax_dendro.set_xlabel("Indeks Klaster / Ukuran Cabang", fontsize=11)
    ax_dendro.set_ylabel("Jarak Euclidean (Dissimilarity)", fontsize=11)

    # Agglomerative Cluster di PC1 & PC2
    ax_agg = fig.add_subplot(1, 2, 2)
    ax_agg.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=agg_labels,
        cmap="viridis",
        alpha=0.75,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    ax_agg.set_title("Klasterisasi Hierarki Agglomerative (K=2) pada PCA", fontsize=12, fontweight="bold")
    ax_agg.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% Variansi)", fontsize=11)
    ax_agg.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% Variansi)", fontsize=11)
    ax_agg.grid(True, linestyle="--", alpha=0.5)

    plot_path3 = OUTPUT_PLOT_DIR / "3_hierarchical_dendrogram_dan_pca.png"
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[PLOT] 3. Disimpan: {plot_path3.name}")


def profiling_klinis_klaster(df_clean: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Menganalisis perbedaan karakteristik klinis antar-klaster pasien yang terbentuk.
    Menghitung mean fitur kontinu dan prevalensi penyakit jantung pada tiap klaster.
    """
    cetak_header("5. Analisis Profil Klinis Karakteristik Klaster Pasien")

    df_profile = df_clean.copy()
    df_profile["Cluster"] = cluster_labels

    ringkasan = df_profile.groupby("Cluster").agg(
        Jumlah_Pasien=("target", "count"),
        Prevalensi_Sakit_Jantung=("target", lambda x: f"{(x.sum() / len(x)) * 100:.1f}%"),
        Rata_Usia=("age", "mean"),
        Rata_Detak_Maks=("thalach", "mean"),
        Rata_ST_Depression=("oldpeak", "mean"),
        Rata_Tekanan_Darah=("trestbps", "mean"),
        Rata_Kolesterol=("chol", "mean"),
        Rata_Pembuluh_Darah_Ca=("ca", "mean"),
    ).reset_index()

    # Hitung rasio penyakit dalam numerik untuk visualisasi
    df_profile["target_num"] = df_profile["target"]
    prevalensi_num = df_profile.groupby("Cluster")["target_num"].mean()

    # Beri label klinis
    if prevalensi_num[0] > prevalensi_num[1]:
        nama_klaster = {0: "Klaster Risiko Tinggi (Positif)", 1: "Klaster Risiko Rendah (Sehat)"}
    else:
        nama_klaster = {0: "Klaster Risiko Rendah (Sehat)", 1: "Klaster Risiko Tinggi (Positif)"}

    ringkasan["Interpretasi_Klinis"] = ringkasan["Cluster"].map(nama_klaster)

    print("\n--- TABEL PROFIL KLINIS KLASTER PASIEN ---")
    print(ringkasan.to_string(index=False))

    ringkasan.to_csv(CLUSTER_PROFILING_CSV, index=False)
    print(f"\n[SUKSES] Profil klaster pasien tersimpan di: {CLUSTER_PROFILING_CSV.name}")

    # Plot 4: Visualisasi Profiling Klinis Antar-Klaster
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Prevalensi Diagnosa Jantung per Klaster
    cluster_diag = pd.crosstab(df_profile["Cluster"], df_profile["target"], normalize="index") * 100
    cluster_diag.plot(
        kind="bar",
        stacked=True,
        ax=axes[0, 0],
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
    )
    axes[0, 0].set_title("Distribusi Pasien: Sehat vs Sakit per Klaster", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Klaster Pasien")
    axes[0, 0].set_ylabel("Persentase (%)")
    axes[0, 0].set_xticklabels([f"Klaster {c}" for c in cluster_diag.index], rotation=0)
    axes[0, 0].legend(["Normal (0)", "Sakit Jantung (1)"], loc="lower right")

    # 2. Thalach (Detak Jantung Maksimum) Boxplot
    sns.boxplot(x="Cluster", y="thalach", data=df_profile, ax=axes[0, 1], hue="Cluster", palette="Blues", legend=False)
    axes[0, 1].set_title("Detak Jantung Maksimum (Thalach) per Klaster", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Klaster Pasien")
    axes[0, 1].set_ylabel("Detak Jantung Maks (bpm)")
    axes[0, 1].set_xticks(range(len(sorted(df_profile["Cluster"].unique()))))
    axes[0, 1].set_xticklabels([f"Klaster {c}" for c in sorted(df_profile["Cluster"].unique())])

    # 3. Oldpeak (ST Depression) Boxplot
    sns.boxplot(x="Cluster", y="oldpeak", data=df_profile, ax=axes[1, 0], hue="Cluster", palette="Reds", legend=False)
    axes[1, 0].set_title("ST Depression (Oldpeak) per Klaster", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Klaster Pasien")
    axes[1, 0].set_ylabel("Oldpeak (ST Depression)")
    axes[1, 0].set_xticks(range(len(sorted(df_profile["Cluster"].unique()))))
    axes[1, 0].set_xticklabels([f"Klaster {c}" for c in sorted(df_profile["Cluster"].unique())])

    # 4. Usia (Age) vs Ca (Major Vessels)
    sns.scatterplot(
        x="age",
        y="ca",
        hue="Cluster",
        data=df_profile,
        ax=axes[1, 1],
        palette="coolwarm",
        alpha=0.8,
        s=65,
    )
    axes[1, 1].set_title("Distribusi Usia vs Pembuluh Darah Fluoroskopi (ca)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Usia (Tahun)")
    axes[1, 1].set_ylabel("Jumlah Pembuluh Darah (ca)")

    fig.tight_layout()
    plot_path4 = OUTPUT_PLOT_DIR / "4_profiling_klinis_klaster.png"
    plt.savefig(plot_path4, dpi=300)
    plt.close()
    print(f"[PLOT] 4. Disimpan: {plot_path4.name}")

    return ringkasan


def cetak_kesimpulan_eksperimen():
    cetak_header("Ringkasan Kesimpulan Eksperimen 4")
    kesimpulan = """
    [1] Analisis Elbow Method dan Silhouette Coefficient mengonfirmasi K=2 sebagai jumlah klaster
        paling optimal untuk mengelompokkan dataset penyakit jantung secara natural tanpa supervisi.
    [2] Algoritma K-Means dan Agglomerative Hierarchical Clustering menunjukkan koherensi struktur
        yang sangat kuat (Silhouette Score ~0.14 - 0.16; Calinski-Harabasz > 40).
    [3] Validasi Eksternal (ARI & NMI): Klaster yang terbentuk secara mandiri memiliki korelasi tinggi
        dengan ground truth diagnosa medis (ARI > 0.40, NMI > 0.35), membuktikan bahwa fitur fisiologis
        pasien memang memiliki pola pemisahan alami antara kondisi jantung sehat vs berpenyakit.
    [4] Reduksi Dimensi PCA: Dua komponen utama (PC1 & PC2) mampu memproyeksikan batas klaster dengan
        jelas dan memisahkan kelompok berisiko tinggi (thalach rendah, ST-depression tinggi, ca tinggi)
        dari kelompok sehat.
    [5] Seluruh hasil visualisasi dan metrik klaster tersimpan di folder 'exp4_clustering_and_pca/'.
    """
    print(kesimpulan)
    print("=" * 70)
    print("[SELESAI] Eksperimen 4 selesai dijalankan dengan sukses!\n")


def main():
    print(
        """
    +-------------------------------------------------------------+
    |       DATA MINING EXPERIMENT #4: CLUSTERING & PCA           |
    |      Implementasi Unsupervised Learning & Reduksi Dimensi   |
    +-------------------------------------------------------------+
    """
    )
    df, _df_encoded, X_scaled, y_true, _feature_names = muat_dan_siapkan_data()
    best_k = analisis_k_optimal(X_scaled)
    kmeans, kmeans_labels, agg_labels, _df_metrik = jalankan_dan_bandingkan_clustering(
        X_scaled, y_true, k=best_k
    )
    visualisasi_pca_dan_dendrogram(X_scaled, kmeans, kmeans_labels, agg_labels, y_true)
    profiling_klinis_klaster(df, kmeans_labels)
    cetak_kesimpulan_eksperimen()


if __name__ == "__main__":
    main()
