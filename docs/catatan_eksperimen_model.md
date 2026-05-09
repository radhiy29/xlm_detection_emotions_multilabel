# Catatan Eksperimen Model Multilabel

Dokumen ini menyimpan ringkasan konfigurasi eksperimen fine-tuning yang sudah pernah dicoba, agar artifact model tambahan dapat dihapus dari folder `models` tanpa kehilangan catatan metodologi.

## Model Final Aplikasi

Model yang digunakan pada aplikasi:

- Nama tampilan: `XLM-RoBERTa Multilabel`
- Path: `models/finetuned-xlm-multilabel-twostage`
- Arsitektur: `xlm-roberta-base`
- Tipe tugas: multilabel emotion classification
- Label: `Senang`, `Sedih`, `Marah`, `Takut`, `Cinta`
- Fungsi loss: `BCEWithLogitsLoss` dengan `pos_weight`
- Aktivasi inferensi: sigmoid
- Strategi training: two-stage fine-tuning

## Rencana Penulisan Bab 3

Pada Bab 3, eksperimen dapat ditulis sebagai rencana pengujian beberapa kombinasi hyperparameter. Hasil akhir dan pemilihan model terbaik baru dijelaskan pada Bab 4.

Tahapan fine-tuning:

1. Stage 1: fine-tuning awal menggunakan dataset tweet emosi berbahasa Indonesia yang diseimbangkan per label.
2. Stage 2: fine-tuning lanjutan menggunakan dataset lirik Hindia multilabel.
3. Validation dan test hanya menggunakan data lirik Hindia agar evaluasi sesuai domain penelitian.

## Kombinasi Hyperparameter Utama

Eksperimen awal menggunakan kombinasi berikut:

| Parameter | Nilai yang Dicoba |
|---|---|
| Tweet per label | `100`, `200`, `300` |
| Stage 2 learning rate | `1e-5`, `5e-6` |

Total kombinasi awal: `3 x 2 = 6` kombinasi.

Konfigurasi lain yang digunakan:

| Parameter | Nilai |
|---|---|
| Base model | `xlm-roberta-base` |
| Stage 1 learning rate | `2e-5` |
| Stage 1 epoch | `2` |
| Stage 2 epoch | `8` |
| Stage 2 batch size | `12` |
| Evaluation batch size | `16` |
| Optimasi GPU | `fp16` jika CUDA tersedia, `tf32` jika tersedia |

## Eksperimen Lanjutan yang Pernah Dicoba

Eksperimen lanjutan pernah dicoba untuk validasi tambahan, tetapi tidak perlu dijelaskan sebagai model terpisah di aplikasi.

Kombinasi validasi lanjutan:

| Parameter | Nilai yang Dicoba |
|---|---|
| Tweet per label | `200`, `300`, `400` |
| Stage 2 learning rate | `1e-5`, `7e-6`, `5e-6` |

Eksperimen tambahan regularisasi:

| Parameter | Nilai yang Dicoba |
|---|---|
| Tweet per label | `300` |
| Stage 2 learning rate | `1e-5` |
| Stage 2 max length | `192`, `256` |
| Stage 2 weight decay | `0.03`, `0.05`, `0.08` |

Catatan: artifact model dari eksperimen lanjutan dapat dihapus karena aplikasi sudah difiksasi hanya memakai `XLM-RoBERTa Multilabel`.

## Threshold Model Final

Threshold multilabel model final:

| Label | Threshold |
|---|---:|
| Senang | `0.30` |
| Sedih | `0.30` |
| Marah | `0.50` |
| Takut | `0.50` |
| Cinta | `0.45` |

## Metrik Model Final

Metrik test model final:

| Metric | Nilai |
|---|---:|
| Micro F1 | `0.6939` |
| Macro F1 | `0.6191` |
| Weighted F1 | `0.7172` |
