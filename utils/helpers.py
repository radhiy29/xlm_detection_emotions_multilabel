REQUIRED_COLUMNS = [
    "judul_lagu",
    "musisi",
    "album",
    "tahun_rilis",
    "lirik_lagu"
]

def validate_dataset(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    if df["lirik_lagu"].isnull().any():
        raise ValueError("Terdapat lirik lagu yang kosong")

    if df.duplicated(subset=["judul_lagu"]).any():
        raise ValueError("Terdapat judul lagu duplikat")