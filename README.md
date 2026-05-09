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

### Performa Per Label

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Senang | 0.588 | - | 0.588 |
| Sedih | 0.842 | - | 0.842 |
| Marah | 0.727 | - | 0.727 |
| Takut | 0.476 | - | 0.476 |
| Cinta | 0.727 | - | 0.727 |

## 🏗️ Struktur Project

```
project_xlm_hindia/
├── analysis/                          # Modul analisis dan training
│   ├── emotion_model.py              # Model inference untuk emosi
│   ├── multilabel_evaluator.py       # Evaluasi model multilabel
│   └── train_model_multilabel_twostage.py  # Script training
├── preprocessing/                     # Preprocessing teks
│   └── text_cleaner.py               # Cleaning lirik
├── utils/                            # Utility functions
│   └── helpers.py                    # Helper functions
├── dataset/                          # Dataset management
│   ├── dataset-emosi-multilabel.csv  # Dataset lirik (ignored)
│   └── emotion_id_opinion.py         # Dataset loader
├── models/                           # Model checkpoints (ignored)
│   ├── experiments-xlm-multilabel-twostage/  # Experiment checkpoints
│   └── finetuned-xlm-multilabel-twostage/    # Final model
├── docs/                             # Dokumentasi
├── app.py                            # Streamlit application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Dokumentasi ini
```

## 🚀 Instalasi

### Prerequisites

- Python 3.8+
- pip
- Git

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
python analysis/train_model_multilabel_twostage.py \
    --base_model xlm-roberta-base \
    --tweet_per_label 300 \
    --stage1_lr 2e-5 \
    --stage2_lr 1e-5 \
    --output_root models/experiments-xlm-multilabel-twostage \
    --final_output_dir models/finetuned-xlm-multilabel-twostage
```

### 3. Evaluasi Model

Evaluasi model dapat dilakukan melalui tab "Evaluasi Model" di aplikasi Streamlit, atau dengan menggunakan modul `multilabel_evaluator.py`.

## 🔬 Metodologi

### Two-Stage Fine-tuning

1. **Stage 1**: Fine-tuning dengan learning rate lebih tinggi (2e-5) untuk adaptasi domain
2. **Stage 2**: Fine-tuning lanjutan dengan learning rate lebih rendah (1e-5) untuk optimasi performa

### Threshold Optimization

Threshold per label ditentukan dari data validasi untuk memaksimalkan F1 score:
- Senang: 0.30
- Sedih: 0.30
- Marah: 0.50
- Takut: 0.50
- Cinta: 0.45

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

## 🎨 Visualisasi

Aplikasi menyediakan berbagai visualisasi:
- Bar chart distribusi emosi per lagu
- Heatmap intensitas emosi per bait
- Radar chart perbandingan emosi antar lagu
- Confusion matrix per label
- Co-occurrence matrix antar emosi

## 🔧 Konfigurasi

### Model Path

Model path dapat dikonfigurasi melalui environment variable:

```bash
export MODEL_PATH="models/finetuned-xlm-multilabel-twostage"
# atau untuk Hugging Face Hub:
export MODEL_PATH="username/xlm-roberta-hindia-emotion"
```

### Streamlit Configuration

Konfigurasi Streamlit dapat diatur di `.streamlit/config.toml` (tidak di-commit).

## 📊 Metrics dan Evaluasi

### Metrics yang Digunakan

- **Micro F1**: Performa keseluruhan pada semua label
- **Macro F1**: Rata-rata performa per label (unweighted)
- **Weighted F1**: Rata-rata performa per label (weighted by support)
- **Precision/Recall per label**: Untuk analisis detail per emosi

### Confusion Matrix

Untuk setiap label, confusion matrix menunjukkan:
- **TP** (True Positive): Prediksi benar positif
- **TN** (True Negative): Prediksi benar negatif
- **FP** (False Positive): Prediksi salah positif
- **FN** (False Negative): Prediksi salah negatif

## 🚧 Limitasi

- Model dilatih pada dataset relatif kecil (344 samples)
- Performa pada label "Takut" masih rendah (F1: 0.476)
- Model spesifik untuk lirik Hindia, mungkin kurang generalize untuk artis lain
- Threshold fixed, tidak adaptive per konteks

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
