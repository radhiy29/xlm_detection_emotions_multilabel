# 🎵 Analisis Emosi Lirik Hindia dengan XLM-RoBERTa

Aplikasi analisis emosi multilabel untuk lirik lagu berbahasa Indonesia menggunakan model XLM-RoBERTa yang di-fine-tune dengan dataset lirik Hindia.

## 📋 Deskripsi Project

Project ini merupakan sistem klasifikasi emosi multilabel yang dapat mengidentifikasi 5 jenis emosi dalam lirik lagu:
- **Senang** (Joy)
- **Sedih** (Sadness)
- **Marah** (Anger)
- **Takut** (Fear)
- **Cinta** (Love)

Model ini menggunakan pendekatan **two-stage fine-tuning** pada XLM-RoBERTa base untuk meningkatkan performa pada dataset lirik berbahasa Indonesia.

## 🎯 Fitur Utama

- ✅ Klasifikasi emosi multilabel (satu lirik bisa memiliki beberapa emosi)
- ✅ Analisis per bait/stanza dengan agregasi skor
- ✅ Evaluasi model dengan metrics lengkap (F1, Precision, Recall)
- ✅ Visualisasi distribusi emosi per lagu dan album
- ✅ Confusion matrix per label emosi
- ✅ Interactive dashboard menggunakan Streamlit

## 📊 Performa Model

| Metric | Score |
|--------|-------|
| Micro F1 | 0.694 |
| Macro F1 | 0.619 |
| Weighted F1 | 0.717 |
| Macro Precision | 0.558 |
| Macro Recall | 0.748 |
| Hamming Loss | 0.257 |
| Subset Accuracy | 0.229 |

Metrik di atas dihitung pada **test set** menggunakan threshold hasil optimasi pada validation set.

### Hasil Optimasi Threshold pada Validation Set

| Label | Threshold | F1 Validasi |
|-------|----------:|------------:|
| Senang | 0.30 | 0.588 |
| Sedih | 0.30 | 0.842 |
| Marah | 0.50 | 0.727 |
| Takut | 0.50 | 0.476 |
| Cinta | 0.45 | 0.727 |

Nilai pada tabel ini bukan precision/recall test set. Threshold dipilih secara terpisah untuk setiap label berdasarkan F1 validation set.

## 🏗️ Struktur Project

```
project_xlm_hindia/
├── analysis/                          # Modul analisis dan training
│   ├── multilabel_evaluator.py       # Evaluasi dan inferensi multilabel aktif
│   └── train_model_multilabel_twostage.py  # Script training
├── preprocessing/                     # Preprocessing teks
│   └── text_cleaner.py               # Cleaning lirik
├── utils/                            # Utility functions
│   └── helpers.py                    # Helper functions
├── dataset/                          # Dataset management
│   ├── dataset-emosi-multilabel.csv  # Dataset training multilabel (tracked)
│   ├── dataset-lirik-lagu-hindia.csv # Dataset lirik mentah (tracked)
│   └── emotion_id_opinion.py         # Referensi loader dataset tweet
├── models/                           # Model checkpoints (ignored)
│   ├── experiments-xlm-multilabel-twostage/  # Experiment checkpoints
│   └── finetuned-xlm-multilabel-twostage/    # Final model
├── docs/                             # Catatan eksperimen model
├── research/                         # Dokumen proposal/skripsi lokal (ignored)
├── .streamlit/                       # Konfigurasi Streamlit (tracked)
├── app.py                            # Streamlit application
├── requirements.txt                  # Python dependencies
├── AGENTS.md                         # Panduan kontribusi project
├── .gitignore                        # Git ignore rules
└── README.md                         # Dokumentasi ini
```

## 🚀 Instalasi

### Prerequisites

- Python 3.8+
- pip
- Git

Lingkungan lokal saat dokumentasi ini diperbarui menggunakan Python 3.13.6. Training penuh dan kompatibilitas CUDA tetap perlu diverifikasi pada mesin yang digunakan karena versi dependency di `requirements.txt` tidak dipin.

### Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd project_xlm_hindia

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

```
streamlit          # Web framework untuk dashboard
pandas             # Data manipulation
numpy              # Numerical operations
torch              # PyTorch untuk deep learning
transformers       # Hugging Face Transformers
scikit-learn       # Machine learning utilities
plotly             # Interactive visualizations
datasets           # Hugging Face Datasets
```

## 🎮 Cara Menggunakan

### 1. Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### 2. Training Model Baru

```bash
python analysis/train_model_multilabel_twostage.py --base_model xlm-roberta-base --tweet_per_label 300 --stage1_lr 2e-5 --output_root models/experiments-xlm-multilabel-twostage --final_output_dir models/finetuned-xlm-multilabel-twostage
```

Pada mode single run, stage 2 menggunakan learning rate `1e-5`, max length `192`, weight decay `0.05`, dan 8 epoch dengan early stopping patience `2`. Untuk menjalankan grid search gunakan flag `--grid`; opsi kandidat learning rate stage 2 diatur melalui `--stage2_lrs`.

Training memerlukan resource yang jauh lebih besar daripada inferensi. GPU CUDA disarankan untuk training; aplikasi dapat melakukan inferensi dengan CPU apabila CUDA tidak tersedia.

### 3. Evaluasi Model

Evaluasi model dapat dilakukan melalui tab "Evaluasi Model" di aplikasi Streamlit, atau dengan menggunakan modul `multilabel_evaluator.py`.

## 🔬 Metodologi

### Two-Stage Fine-tuning

1. **Stage 1 — adaptasi bahasa Indonesia**: Fine-tuning pada subset seimbang [Emotion Dataset from Indonesian Public Opinion](https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion), dengan learning rate `2e-5`, 2 epoch, dan max length `128`.
2. **Stage 2 — spesialisasi lirik Hindia**: Fine-tuning lanjutan pada dataset lirik Hindia multilabel, dengan learning rate `1e-5`, maksimal 8 epoch, max length `192`, weight decay `0.05`, dan early stopping patience `2`.

Training menggunakan `BCEWithLogitsLoss` dengan `pos_weight` per label yang dihitung dari training set. Oversampling label langka (`Takut` dan `Senang`) diaktifkan secara default. Pemilihan checkpoint stage 2 menggunakan macro F1 validation set.

### Threshold Optimization

Threshold per label ditentukan dari data validasi untuk memaksimalkan F1 score:
- Senang: 0.30
- Sedih: 0.30
- Marah: 0.50
- Takut: 0.50
- Cinta: 0.45

Pencarian dilakukan pada rentang `0.20`–`0.75` dengan step `0.05`. Test set tidak digunakan untuk memilih threshold. Jika tidak ada skor yang melewati threshold saat inferensi, label dengan probabilitas tertinggi dipakai sebagai fallback.

### Agregasi Skor Bait

Untuk lirik panjang yang dibagi menjadi beberapa bait:
```
final_score = (0.7 × mean_score) + (0.3 × max_score)
```

## 📝 Dataset

Dataset terdiri dari lirik lagu Hindia yang dianotasi dengan label emosi multilabel:
- **Total samples**: 344
- **Train**: 275 (80%)
- **Validation**: 34 (10%)
- **Test**: 35 (10%)

Dataset menggunakan stratified split berdasarkan label emosi paling jarang untuk menjaga distribusi.

Split dibuat dengan rasio `80/10/10` dan `random_state=42`. Nilai ini dipertahankan untuk menjaga reproduktibilitas hasil yang dilaporkan.

### Skema Dataset Training

Kolom pada `dataset/dataset-emosi-multilabel.csv`:

| Kolom | Keterangan |
|-------|------------|
| `judul_lagu` | Judul lagu |
| `musisi` | Nama musisi |
| `album` | Nama album |
| `tahun_rilis` | Tahun rilis |
| `lirik_lagu` | Teks lirik |
| `label_emosi` | Satu atau beberapa label yang dipisahkan koma, misalnya `Sedih, Cinta` |

Label yang valid adalah `Senang`, `Sedih`, `Marah`, `Takut`, dan `Cinta`. `dataset/dataset-emosi-multilabel.csv` adalah dataset yang dipakai untuk training dan evaluasi, sedangkan `dataset/dataset-lirik-lagu-hindia.csv` menyimpan kumpulan lirik mentah. Kedua file CSV tersebut dilacak oleh Git.

Untuk upload melalui tab **Input Data**, kolom yang wajib hanya `judul_lagu`, `musisi`, `album`, `tahun_rilis`, dan `lirik_lagu`. Kolom `label_emosi` tidak diperlukan untuk inferensi.

## 🎨 Visualisasi

Aplikasi menyediakan berbagai visualisasi:
- Radar skor emosi untuk lagu yang dipilih
- Tren skor emosi per bait
- Distribusi jumlah emosi per lagu dan frekuensi emosi diskografi
- Heatmap emosi per album
- Distribusi emosi per album
- Kombinasi/co-occurrence emosi
- Tren emosi berdasarkan tahun rilis
- Performa dan confusion matrix per label pada test set

## 🔧 Konfigurasi

### Model Path

Model aktif harus tersedia pada path relatif berikut:

```text
models/finetuned-xlm-multilabel-twostage
```

Saat ini aplikasi menggunakan path tersebut secara langsung dan belum membaca environment variable `MODEL_PATH`. Loading langsung dari Hugging Face Hub juga belum menjadi contract aplikasi karena threshold dibaca dari `multilabel_metadata.json` lokal.

Model tidak disertakan dalam Git karena ukurannya sekitar 1 GB. Sebelum menjalankan aplikasi, salin model final beserta tokenizer dan `multilabel_metadata.json` ke path di atas, peroleh model dari pemilik project, atau jalankan kembali script training.

### Streamlit Configuration

Konfigurasi tema dan server Streamlit disimpan di `.streamlit/config.toml` dan dilacak oleh Git. Secret lokal tetap harus disimpan di `.streamlit/secrets.toml`, yang diabaikan oleh Git.

## 📊 Metrics dan Evaluasi

### Metrics yang Digunakan

- **Micro F1**: Performa keseluruhan pada semua label
- **Macro F1**: Rata-rata performa per label (unweighted)
- **Weighted F1**: Rata-rata performa per label (weighted by support)
- **Precision/Recall per label**: Untuk analisis detail per emosi
- **Hamming Loss**: Proporsi keputusan label yang salah dari seluruh pasangan sampel-label
- **Subset Accuracy**: Proporsi sampel yang seluruh kombinasi labelnya diprediksi tepat

### Confusion Matrix

Untuk setiap label, confusion matrix menunjukkan:
- **TP** (True Positive): Prediksi benar positif
- **TN** (True Negative): Prediksi benar negatif
- **FP** (False Positive): Prediksi salah positif
- **FN** (False Negative): Prediksi salah negatif

## 🚧 Limitasi

- Model dilatih pada dataset relatif kecil (344 samples)
- F1 validasi label "Takut" saat tuning threshold masih rendah (0.476)
- Model spesifik untuk lirik Hindia, mungkin kurang generalize untuk artis lain
- Threshold tetap per label dan tidak adaptif terhadap konteks
- Dependency tidak dipin sehingga hasil instalasi dapat berbeda antar waktu dan lingkungan

## 🔮 Future Work

- [ ] Augmentasi data untuk meningkatkan ukuran dataset
- [ ] Experiment dengan model yang lebih besar (XLM-RoBERTa large)
- [ ] Implementasi adaptive threshold
- [ ] Support untuk artis Indonesia lainnya
- [ ] API endpoint untuk inference
- [ ] Deploy ke Hugging Face Spaces

## 📄 Lisensi

[Tentukan lisensi project Anda]

## 👥 Kontributor

- [Nama Anda] - Initial work

## 🙏 Acknowledgments

- Dataset lirik dari Hindia
- Hugging Face untuk Transformers library
- XLM-RoBERTa model dari Facebook AI

## 📧 Kontak

Untuk pertanyaan atau feedback, silakan hubungi [email Anda] atau buat issue di repository ini.

---

**Note**: Model weights tidak disertakan dalam repository ini karena ukurannya yang besar (~1 GB). Untuk menggunakan model, silakan download dari [Hugging Face Hub] atau train ulang menggunakan script yang disediakan.
