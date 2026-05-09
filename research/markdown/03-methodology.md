# BAB III METODOLOGI PENELITIAN

## 3.3 Tahapan Penelitian

### 3.3.1 Identifikasi Masalah
Penelitian ini diawali dengan identifikasi permasalahan terkait subjektivitas dalam analisis emosi pada lirik lagu serta keterbatasan pendekatan konvensional dalam mengolah teks berskala besar.

### 3.3.2 Studi Pustaka
Studi pustaka dilakukan untuk memahami konsep, metode, dan penelitian terdahulu yang relevan, khususnya terkait NLP, Transformer, dan XLM-RoBERTa.

### 3.3.3 Pengumpulan Data
Data terdiri dari:
- Data primer: 54 lagu Hindia (2018–2025), dipecah per bait, dilabeli manual, dan divalidasi expert
- Data sekunder: 7.080 tweet berlabel emosi sebagai data pendukung

### 3.3.4 Preprocessing Data
- Cleaning teks
- Normalisasi terbatas
- Penyeragaman label

### 3.3.5 Pemodelan
Menggunakan model XLM-RoBERTa dengan proses fine-tuning.

### 3.3.6 Evaluasi Model
Menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

### 3.3.7 Analisis dan Visualisasi
Hasil klasifikasi digunakan untuk analisis dan divisualisasikan dengan Streamlit.

### 3.3.8 Penyusunan Laporan
Hasil penelitian disusun dalam bentuk skripsi.

---

## 3.4 Proses Machine Learning

### 3.4.1 Persiapan Dataset
Dataset terdiri dari:
- Lirik Hindia (primer, multi-label per bait)
- Tweet (sekunder)

### 3.4.2 Skema Pelabelan Emosi
Menggunakan pendekatan multi-label:
- Senang
- Sedih
- Marah
- Takut
- Cinta

Satu bait dapat memiliki lebih dari satu label.

### 3.4.3 Pembagian Data
Dataset dibagi menjadi training, validation, dan testing.

### 3.4.4 Tokenisasi
Menggunakan tokenizer XLM-RoBERTa (SentencePiece).

### 3.4.5 Fine-Tuning Model
Model XLM-RoBERTa dengan layer klasifikasi dilatih menggunakan dataset.

### 3.4.6 Fungsi Loss dan Optimasi
- Loss: Binary Cross-Entropy
- Optimizer: AdamW

### 3.4.7 Evaluasi Model

Meskipun model dilatih menggunakan pendekatan multi-label, evaluasi dilakukan dengan mengonversi output probabilitas menjadi satu label dominan berdasarkan nilai probabilitas tertinggi.

Langkah evaluasi:
1. Model menghasilkan probabilitas untuk setiap kategori emosi
2. Dipilih satu label dengan probabilitas tertinggi sebagai label dominan
3. Label dominan digunakan untuk:
   - Confusion matrix
   - Accuracy
   - Precision
   - Recall
   - F1-score

Pendekatan ini digunakan untuk mempermudah interpretasi performa model menggunakan metode evaluasi klasifikasi konvensional.

---

## 3.5 Proses Visualisasi

### 3.5.1 Tujuan Visualisasi
Menyajikan hasil klasifikasi emosi dalam bentuk visual.

### 3.5.2 Sumber Data
- Label dominan (untuk pola umum)
- Probabilitas emosi (untuk analisis mendalam)

### 3.5.3 Jenis Visualisasi
- Pie chart (distribusi emosi dominan)
- Line chart (tren emosi)
- Radar chart (profil emosi per lagu)
- Bar chart (confidence score)
- Heatmap (emosi per lagu/album)

### 3.5.4 Implementasi
Menggunakan Streamlit.

### 3.5.5 Batasan
Visualisasi bergantung pada output model dan tidak merepresentasikan makna absolut.
