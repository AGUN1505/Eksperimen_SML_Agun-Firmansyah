"""
automate_Agun-Firmansyah.py
============================
Script otomatisasi preprocessing Heart Disease Dataset.
Mengkonversi langkah-langkah eksperimen dari notebook menjadi
fungsi yang dapat dijalankan secara otomatis.

Cara penggunaan:
    python automate_Agun-Firmansyah.py
    python automate_Agun-Firmansyah.py --input heart.csv --output heart-disease_preprocessing.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame hasil loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[LOAD] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ─────────────────────────────────────────────
# 2. HAPUS DUPLIKAT
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa duplikat.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"[DUPLIKAT] Dihapus: {before - after} baris | Sisa: {after} baris")
    return df


# ─────────────────────────────────────────────
# 3. TANGANI MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values:
    - Fitur numerik  → diisi dengan median
    - Fitur kategorikal → diisi dengan modus

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    numerical_cols   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    total_missing = df.isnull().sum().sum()

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [MISSING] {col}: diisi median = {median_val:.2f}")

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [MISSING] {col}: diisi modus = {mode_val}")

    print(f"[MISSING] Ditangani: {total_missing} nilai | Sisa: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 4. ENCODING KATEGORIKAL
# ─────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encoding untuk fitur kategorikal multi-kelas.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame setelah encoding.
    """
    multi_class_cols = ['cp', 'restecg', 'slope', 'thal']
    cols_to_encode   = [c for c in multi_class_cols if c in df.columns]

    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode, drop_first=False)
    after_cols = df.shape[1]

    print(f"[ENCODING] One-Hot Encoding pada: {cols_to_encode}")
    print(f"[ENCODING] Kolom: {before_cols} → {after_cols}")
    return df


# ─────────────────────────────────────────────
# 5. TANGANI OUTLIER (IQR CLIPPING)
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Clipping.
    Nilai di luar batas [Q1 - 1.5*IQR, Q3 + 1.5*IQR] akan di-clip.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame setelah penanganan outlier.
    """
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cols_to_clip   = [c for c in numerical_cols if c in df.columns]

    total_outliers = 0
    for col in cols_to_clip:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        total_outliers += n_outliers
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  [OUTLIER] {col}: {n_outliers} outlier di-clip ke [{lower:.2f}, {upper:.2f}]")

    print(f"[OUTLIER] Total outlier ditangani: {total_outliers}")
    return df


# ─────────────────────────────────────────────
# 6. NORMALISASI
# ─────────────────────────────────────────────
def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi fitur numerik menggunakan StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame setelah normalisasi.
    """
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cols_to_scale  = [c for c in numerical_cols if c in df.columns]

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print(f"[NORMALISASI] StandardScaler diterapkan pada: {cols_to_scale}")
    return df


# ─────────────────────────────────────────────
# 7. SIMPAN HASIL
# ─────────────────────────────────────────────
def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Args:
        df (pd.DataFrame): DataFrame yang akan disimpan.
        output_path (str): Path output file CSV.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Dataset disimpan ke: {output_path} | Shape: {df.shape}")


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline lengkap preprocessing Heart Disease Dataset.

    Args:
        input_path  (str): Path ke dataset mentah (.csv).
        output_path (str): Path output dataset hasil preprocessing.

    Returns:
        pd.DataFrame: Dataset siap latih.
    """
    print("=" * 55)
    print("   PIPELINE PREPROCESSING - Heart Disease Dataset")
    print("=" * 55)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Hapus duplikat
    df = remove_duplicates(df)

    # Step 3: Tangani missing values
    df = handle_missing_values(df)

    # Step 4: Encoding kategorikal
    df = encode_categorical(df)

    # Step 5: Tangani outlier
    df = handle_outliers(df)

    # Step 6: Normalisasi
    df = normalize_features(df)

    # Step 7: Simpan
    save_data(df, output_path)

    print("=" * 55)
    print("   PREPROCESSING SELESAI!")
    print(f"   Input  : {input_path}")
    print(f"   Output : {output_path}")
    print(f"   Shape  : {df.shape}")
    print("=" * 55)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing Heart Disease Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="heart.csv",
        help="Path ke dataset mentah (default: heart.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preprocessing/heart-disease_preprocessing.csv",
        help="Path output dataset preprocessing"
    )
    args = parser.parse_args()

    run_preprocessing(input_path=args.input, output_path=args.output)
