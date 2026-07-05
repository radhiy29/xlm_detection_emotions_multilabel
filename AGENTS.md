# Analisis Emosi Lirik Hindia — Agent Guide

## 1. Non-Negotiable Rules

- Push back kalau ada yang aneh. Kalau arsitektur model, pipeline preprocessing, skema dataset, atau pendekatan evaluasi terasa bermasalah, sebutkan dan diskusikan dulu sebelum implementasi.
- Klarifikasi sebelum implementasi besar. Kalau requirement ambigu (misal: "ubah threshold" bisa berarti banyak hal), tanya dulu daripada menulis banyak kode ke arah yang salah.
- Stick ke scope. Jangan refactor modul/file di luar scope task. Kalau melihat masalah lain, sebutkan di akhir response sebagai catatan.
- Posisikan diri sebagai kontributor, bukan arsitek. Ikuti konvensi yang sudah ada. Jangan introduce library, abstraction, atau dependency baru tanpa diskusi.
- Jangan print, expose, atau edit secret/credential kecuali user secara eksplisit meminta. Kalau perlu membahas env, redaksi nilai sensitif.

### 1.1 Project-Specific Important Rules

- **Model weights adalah aset besar (~1 GB)**. Jangan commit model ke Git. Semua file model sudah di-`.gitignore`. Kalau perlu membahas path model, gunakan path relatif `models/finetuned-xlm-multilabel-twostage`.
- **Dataset CSV di-track di Git** (`dataset/*.csv`), termasuk dataset training multilabel dan dataset lirik mentah. File data besar lain harus ditempatkan di `data/` agar di-ignore. Jangan mengubah struktur kolom dataset tanpa diskusi karena mempengaruhi training, evaluasi, dan aplikasi.
- **Threshold per label ditentukan dari data validasi**, bukan hardcoded sembarangan. Kalau ada perubahan threshold, harus melalui proses tuning ulang atau diskusi eksplisit.
- **Jangan mengubah split ratio atau random state** tanpa diskusi — ini mempengaruhi reproduktibilitas hasil yang sudah dilaporkan di skripsi.
- **Catatan eksperimen** di `docs/catatan_eksperimen_model.md` harus di-update kalau ada perubahan konfigurasi training atau hasil model baru.
- Kalau ada test, command, atau verifikasi yang tidak bisa kamu jalankan (misal: training GPU), minta user menjalankannya manual via terminal dan sebutkan command/context yang diperlukan.
- **Konteks skripsi**: Project ini adalah project skripsi. Perubahan harus mempertimbangkan konsistensi dengan dokumen proposal/skripsi di `research/`.

---

## 2. Source of Truth

**Model final** di `models/finetuned-xlm-multilabel-twostage` beserta `multilabel_metadata.json` adalah source of truth untuk behavior prediksi dan konfigurasi model aktif.

Prioritas keputusan:

1. `multilabel_metadata.json` — threshold, label, metrik, konfigurasi training yang menghasilkan model final
2. Script training `analysis/train_model_multilabel_twostage.py` — pipeline training yang reproducible
3. Evaluator `analysis/multilabel_evaluator.py` — pipeline evaluasi dan inferensi yang dipakai aplikasi
4. Aplikasi Streamlit `app.py` — UI yang mengkonsumsi model dan menampilkan hasil

Kalau ada perubahan behavior atau contract:

- ubah modul `analysis/` terlebih dulu
- sesuaikan `app.py` bila perlu
- update `multilabel_metadata.json` bila konfigurasi model berubah
- update `docs/catatan_eksperimen_model.md` untuk dokumentasi eksperimen

---

## 3. Read Before Implementing

Sebelum membuat modul/fitur baru, baca modul reference yang mirip:

- **Training pipeline**: baca `analysis/train_model_multilabel_twostage.py` untuk konvensi training.
- **Evaluasi & inferensi**: baca `analysis/multilabel_evaluator.py` untuk konvensi prediksi dan metrik.
- **Preprocessing teks**: baca `preprocessing/text_cleaner.py`.
- **UI Streamlit**: baca `app.py` untuk konvensi layout, styling, dan komponen.
- **Dataset handling**: baca `dataset/emotion_id_opinion.py` dan `utils/helpers.py`.

Konsistensi dengan codebase aktif lebih penting daripada "cara yang lebih bersih" secara abstrak.

---

## 4. Project Structure

```text
project_xlm_hindia/
├── analysis/                              # Modul analisis, training, dan evaluasi
│   ├── __init__.py
│   ├── multilabel_evaluator.py           # Evaluasi dan inferensi multilabel (aktif)
│   └── train_model_multilabel_twostage.py # Script training two-stage
├── preprocessing/                         # Preprocessing teks
│   ├── __init__.py
│   └── text_cleaner.py                   # Cleaning lirik lagu
├── utils/                                 # Utility functions
│   ├── __init__.py
│   └── helpers.py                        # Validasi dataset
├── dataset/                               # Dataset management
│   ├── dataset-emosi-multilabel.csv      # Dataset lirik Hindia (tracked)
│   ├── dataset-lirik-lagu-hindia.csv     # Dataset lirik mentah (tracked)
│   └── emotion_id_opinion.py             # NusaCrowd dataset loader (tweet)
├── models/                                # Model checkpoints (ignored, ~1 GB+)
│   ├── experiments-xlm-multilabel-twostage/  # Experiment grid search
│   │   ├── stage1_tweet{N}_lr{X}/        # Checkpoint stage 1
│   │   ├── final_tweet{N}_lr{X}/         # Checkpoint stage 2
│   │   └── grid_results.json             # Hasil grid search
│   └── finetuned-xlm-multilabel-twostage/   # Model final (aktif)
│       ├── config.json                   # HuggingFace model config
│       ├── model.safetensors             # Weights (~1.1 GB)
│       ├── multilabel_metadata.json      # Threshold, metrik, konfigurasi
│       ├── tokenizer.json                # Tokenizer XLM-RoBERTa
│       ├── tokenizer_config.json
│       └── training_args.bin
├── docs/                                  # Dokumentasi
│   └── catatan_eksperimen_model.md       # Catatan eksperimen training
├── research/                              # Dokumen skripsi (ignored)
│   ├── f1e122182_proposal skripsi.pdf
│   ├── revisi setelah sempro.pdf
│   └── markdown/                         # Draft skripsi dalam markdown
│       ├── 00-overview.md
│       ├── 01-introduction.md
│       ├── 02-literature.md
│       ├── 03-methodology.md
│       ├── 04-references.md
│       └── BAB_3_METODOLOGI_PENELITIAN_LENGKAP.md
├── .streamlit/                            # Streamlit configuration
│   └── config.toml                       # Theme dan server settings
├── app.py                                 # Aplikasi Streamlit utama
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Dokumentasi project
```

---

## 5. Model dan Training

### 5.1 Arsitektur Model

- Base model: `xlm-roberta-base` (XLM-RoBERTa dari Facebook AI)
- Task: multilabel emotion classification (5 label)
- Label: `Senang`, `Sedih`, `Marah`, `Takut`, `Cinta`
- Loss function: `BCEWithLogitsLoss` dengan `pos_weight` per label
- Aktivasi inferensi: sigmoid
- Framework: PyTorch + Hugging Face Transformers

### 5.2 Two-Stage Fine-Tuning

Model dilatih dengan pendekatan two-stage:

1. **Stage 1**: Fine-tuning pada dataset tweet emosi berbahasa Indonesia yang diseimbangkan per label
   - Sumber: [Emotion Dataset from Indonesian Public Opinion](https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion)
   - Tujuan: adaptasi domain bahasa Indonesia
   - Learning rate: `2e-5`
   - Epoch: `2`
   - Max length: `128`
   - Metric: `eval_loss` (minimize)

2. **Stage 2**: Fine-tuning lanjutan pada dataset lirik Hindia multilabel
   - Tujuan: spesialisasi domain lirik lagu
   - Learning rate: `1e-5`
   - Epoch: `8` (dengan early stopping patience `2`)
   - Max length: `192`
   - Weight decay: `0.05`
   - Metric: `f1_macro` (maximize)
   - `pos_weight` dihitung dari distribusi label training set

### 5.3 Dataset

#### Dataset Lirik Hindia (`dataset/dataset-emosi-multilabel.csv`)

- Kolom wajib: `judul_lagu`, `musisi`, `album`, `tahun_rilis`, `lirik_lagu`, `label_emosi`
- Label multilabel dipisahkan koma (misal: `Sedih, Cinta`)
- Total samples: **344**
- Split: train **275** (80%) | validation **34** (10%) | test **35** (10%)
- Stratified split berdasarkan label emosi paling jarang per sample
- Random state: `42` (jangan diubah tanpa diskusi)

#### Dataset Tweet (Stage 1)

- Sumber: CSV dari GitHub repository Ricco48
- Diunduh langsung saat training via URL
- Label: anger, fear, joy, love, sad (di-mapping ke label Indonesia)
- Per label dipilih `tweet_per_label` sample (default `300`)
- Filter: panjang 3–60 kata, deduplicated

### 5.4 Threshold Optimization

Threshold per label ditentukan dari data validasi lirik Hindia:

| Label  | Threshold | F1 Validasi |
|--------|-----------|-------------|
| Senang | 0.30      | 0.588       |
| Sedih  | 0.30      | 0.842       |
| Marah  | 0.50      | 0.727       |
| Takut  | 0.50      | 0.476       |
| Cinta  | 0.45      | 0.727       |

- Range pencarian: `0.20` sampai `0.75` dengan step `0.05`
- Kalau semua label di bawah threshold (tidak ada prediksi), fallback ke argmax probability

### 5.5 Model Performance (Test Set)

| Metrik           | Nilai  |
|------------------|--------|
| Micro F1         | 0.6939 |
| Macro F1         | 0.6191 |
| Weighted F1      | 0.7172 |
| Macro Precision  | 0.5579 |
| Macro Recall     | 0.7476 |
| Hamming Loss     | 0.2571 |
| Subset Accuracy  | 0.2286 |

### 5.6 Training Conventions

- Script training menggunakan `argparse` untuk semua hyperparameter.
- Grid search bisa diaktifkan dengan flag `--grid`.
- Text cleaning harus konsisten antara training (`clean_text` di training script) dan inferensi (`clean_lyrics` di `text_cleaner.py`).
- Model dan tokenizer disimpan bersama di directory output.
- Metadata JSON (`multilabel_metadata.json`) selalu disimpan bersama model final.
- GPU: otomatis deteksi CUDA, support `fp16` dan `tf32`.
- Oversampling label rare (`Takut`, `Senang`) diaktifkan secara default.

---

## 6. Pipeline Inferensi

### 6.1 Inferensi Multilabel (Aktif)

Pipeline inferensi yang dipakai aplikasi ada di `analysis/multilabel_evaluator.py`:

1. **Load model**: `load_multilabel_model()` — load tokenizer, model, threshold dari metadata
2. **Prediksi batch**: `predict_multilabel_texts()` — tokenize, forward pass, sigmoid, thresholding
3. **Split bait**: `split_lyrics_to_stanzas()` — pecah lirik menjadi bait berdasarkan blank line
4. **Agregasi skor bait**: `aggregate_stanza_scores()` — rumus: `(0.7 × mean) + (0.3 × max)`
5. **Konversi label**: `labels_from_vector()`, `dominant_label_from_probs()`

### 6.2 Aturan Prediksi

- Tokenisasi: `AutoTokenizer`, max length `192`, truncation `True`, padding `True`
- Batch size inferensi: `16`
- Kalau semua label di bawah threshold, argmax probability diambil sebagai label tunggal
- Lirik panjang dipecah per bait; bait > `120` kata dipecah lagi per chunk
- Skor final per lagu = agregasi skor semua bait

### 6.3 Model Caching

- Model di-cache oleh Streamlit via `@st.cache_resource`
- Data split di-cache via `@st.cache_data`
- Jangan panggil `load_multilabel_model()` berulang di luar cache — model ~1 GB

---

## 7. Aplikasi Streamlit

### 7.1 Entry Point

- File utama: `app.py`
- Jalankan: `streamlit run app.py`
- Default port: `8501`
- Config: `.streamlit/config.toml`

### 7.2 Tab Aplikasi

Aplikasi memiliki 3 tab:

1. **Evaluasi Model** — Evaluasi model pada test set lirik Hindia
   - Menampilkan: metrik ringkasan, performa per label, confusion matrix
   - Data evaluasi dari `load_lyrics_multilabel_splits()`

2. **Input Data** — Upload dataset CSV lirik lagu
   - Kolom wajib: `judul_lagu`, `musisi`, `album`, `tahun_rilis`, `lirik_lagu`
   - Validasi via `validate_dataset()`
   - Preview 20 baris pertama

3. **Analisis Emosi** — Prediksi emosi per bait dan per lagu
   - Menampilkan: tabel prediksi, radar chart, tren per bait, rincian skor per bait
   - Diskografi: distribusi emosi, heatmap album, co-occurrence, tren tahunan

### 7.3 UI Conventions

- Warna emosi konsisten di seluruh aplikasi (didefinisikan di `EMOTION_COLORS`):
  - Senang: `#DDA15E`, Sedih: `#6F8FAF`, Marah: `#C97064`, Takut: `#8E7AB5`, Cinta: `#C88FA3`
- Heatmap scale: cream → coklat tua (`HEATMAP_SCALE`)
- Tabel HTML custom dengan class CSS, bukan `st.dataframe()`
- Download CSV tersedia untuk setiap tabel via `download_table()`
- Setiap section pakai `section_title(icon, text)` untuk heading konsisten
- Setiap chart punya `chart_note()` untuk penjelasan cara membaca
- Session state dipakai untuk menyimpan hasil evaluasi dan analisis antar tab
- Session key di-namespace per model via `model_session_key()`

### 7.4 Styling

- Custom CSS diterapkan via `st.markdown(unsafe_allow_html=True)` di awal `app.py`
- Style meliputi: metric cards, tabel, download button, prediction table, stanza table
- Tema Streamlit di `.streamlit/config.toml`: light base, sans serif font
- Charting pakai Plotly (`plotly.express` dan `plotly.graph_objects`)
- Jangan mengubah `EMOTION_COLORS`, `CHART_COLORS`, atau `HEATMAP_SCALE` tanpa diskusi — konsistensi visual penting

---

## 8. Preprocessing

### 8.1 Text Cleaning (`preprocessing/text_cleaner.py`)

Fungsi `clean_lyrics()`:
1. Hapus penanda struktur lagu: `[Verse]`, `[Chorus]`, `[Bridge]`, dll.
2. Normalisasi newline → spasi
3. Normalisasi spasi berlebih
4. Trim whitespace

### 8.2 Stanza Splitting (`analysis/multilabel_evaluator.py`)

Fungsi `split_lyrics_to_stanzas()`:
1. Normalize line endings
2. Hapus tag `[...]`
3. Split berdasarkan blank line ganda
4. Fallback ke split per line kalau tidak ada blank line
5. Clean setiap stanza via `clean_lyrics()`
6. Chunk stanza > `120` kata

### 8.3 Dataset Validation (`utils/helpers.py`)

Fungsi `validate_dataset()`:
- Kolom wajib: `judul_lagu`, `musisi`, `album`, `tahun_rilis`, `lirik_lagu`
- Tidak boleh ada `lirik_lagu` null
- Tidak boleh ada `judul_lagu` duplikat

---

## 9. Dependencies

```text
streamlit          # Web framework dashboard
pandas             # Data manipulation
numpy              # Numerical operations
torch              # PyTorch deep learning
transformers       # Hugging Face Transformers
scikit-learn       # Metrics (F1, precision, recall, confusion matrix)
plotly             # Interactive visualizations
datasets           # Hugging Face Datasets (untuk training)
```

### 9.1 Aturan Dependencies

- Semua dependency didaftarkan di `requirements.txt` tanpa versi pin.
- Jangan tambah dependency baru tanpa diskusi.
- PyTorch harus kompatibel dengan versi CUDA yang tersedia di mesin user (kalau GPU).
- `nusacrowd` hanya dipakai di `dataset/emotion_id_opinion.py` dan tidak terdaftar di `requirements.txt` — file ini adalah referensi, bukan dependency utama.

---

## 10. Conventions

### 10.1 Python Coding Style

- Python files memakai **spaces** (bukan tabs).
- Import dikelompokkan: stdlib → third-party → local, dipisahkan blank line.
- Type hints dipakai di fungsi signature utama.
- Docstring dalam bahasa Indonesia untuk modul yang langsung dipakai user.
- Variable dan kolom DataFrame biasanya dalam bahasa Indonesia (sesuai domain skripsi).
- Constant pakai UPPER_SNAKE_CASE.
- Fungsi utility tetap kecil dan fokus satu tanggung jawab.

### 10.2 Git dan File Management

- **Git tracked**: source code, dataset CSV, docs, requirements, config.
- **Git ignored**: model weights, `data/`, `research/`, cache, env files, notebook, IDE files.
- Jangan menambah file besar ke Git (model, dataset besar, gambar berat).
- File `research/` hanya ada lokal — berisi dokumen skripsi (proposal, revisi).

### 10.3 Numerical Conventions

- Probability/confidence score: `float32`, range `[0, 1]`, display `:.2f` atau `:.4f`
- Threshold: `float32`, range `[0.2, 0.75]`
- Metrik (F1, precision, recall): display `:.2%` di UI, `:.4f` di metadata
- Jumlah/count: `int`
- Label vector: `int` (`0` atau `1`), shape `(n_samples, 5)`

---

## 11. Workflow Checklists

### 11.1 Mengubah Model atau Training

- [ ] Baca `analysis/train_model_multilabel_twostage.py` dan `docs/catatan_eksperimen_model.md`
- [ ] Pastikan perubahan tidak merusak reproduktibilitas hasil yang sudah dilaporkan
- [ ] Jalankan training ulang (minta user kalau butuh GPU)
- [ ] Update `multilabel_metadata.json` dengan threshold dan metrik baru
- [ ] Update `docs/catatan_eksperimen_model.md` dengan catatan eksperimen
- [ ] Update `README.md` bagian performa model kalau metrik berubah signifikan
- [ ] Verifikasi evaluasi dan prediksi di aplikasi Streamlit masih berfungsi

### 11.2 Mengubah Pipeline Inferensi

- [ ] Baca `analysis/multilabel_evaluator.py`
- [ ] Pastikan perubahan konsisten antara training dan inferensi
- [ ] Verifikasi `predict_multilabel_texts()` masih menghasilkan output yang benar
- [ ] Verifikasi `aggregate_stanza_scores()` masih menghasilkan skor agregasi yang benar
- [ ] Test di aplikasi Streamlit: tab Evaluasi dan tab Analisis Emosi

### 11.3 Mengubah UI/Streamlit

- [ ] Baca `app.py` dan pahami tab yang terkait
- [ ] Ikuti style CSS yang sudah ada — jangan campur styling approach
- [ ] Pastikan session state key konsisten via `model_session_key()`
- [ ] Pastikan setiap tabel punya download CSV
- [ ] Pastikan setiap chart punya `chart_note()` penjelasan
- [ ] Verifikasi layout responsif di browser

### 11.4 Mengubah Dataset

- [ ] Baca `utils/helpers.py` untuk kolom wajib
- [ ] Pastikan kolom `label_emosi` menggunakan nama label Indonesia: `Senang`, `Sedih`, `Marah`, `Takut`, `Cinta`
- [ ] Pastikan tidak ada judul duplikat atau lirik null
- [ ] Kalau dataset berubah signifikan, training harus diulang
- [ ] Update split sizes di `README.md` dan `multilabel_metadata.json`

### 11.5 Mengubah Preprocessing

- [ ] Baca `preprocessing/text_cleaner.py`
- [ ] Pastikan perubahan diterapkan konsisten di `clean_lyrics()` dan `clean_text()` (training)
- [ ] Kalau preprocessing berubah, training harus diulang karena distribusi data berubah
- [ ] Verifikasi stanza splitting masih bekerja dengan benar

---

## 12. Limitasi dan Catatan

- Model dilatih pada dataset relatif kecil (344 samples lirik Hindia)
- Performa pada label `Takut` masih rendah (F1: 0.476)
- Model spesifik untuk lirik Hindia, mungkin kurang generalize untuk artis lain
- Threshold fixed, tidak adaptive per konteks
- File `dataset/emotion_id_opinion.py` adalah NusaCrowd-compatible loader untuk dataset tweet — bukan dataset utama project
