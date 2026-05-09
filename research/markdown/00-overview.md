# Overview Proposal

## Identitas

- Judul: Analisis Emosi pada Lirik Lagu Hindia Menggunakan Model XLM-Roberta dan Aplikasi Visualisasi Berbasis Streamlit.
- Jenis dokumen: Proposal skripsi.
- Program studi: Sistem Informasi, Fakultas Sains dan Teknologi, Universitas Jambi.
- Periode pelaksanaan yang ditulis: November 2025 sampai April 2026.

## Peta Isi Utama

- I. Pendahuluan
- II. Tinjauan Pustaka
- III. Metodologi Penelitian
- Daftar Pustaka

## Fokus Ilmiah Proposal

- Objek: 54 lirik lagu Hindia (main artist) yang tersedia di Spotify.
- Tujuan teknis: klasifikasi dan pemetaan emosi secara otomatis dan terukur (kuantitatif).
- Model inti: XLM-RoBERTa (analisis emosi — fine-tuning + evaluasi kuantitatif), BERTopic berbasis embedding (analisis tema — pendukung).
- Evaluasi: accuracy, precision, recall, F1-score, dan confusion matrix.
- Implementasi sistem: aplikasi visualisasi interaktif dengan Streamlit.

## Perubahan Signifikan dari Versi Sebelumnya

- Judul berubah: "Analisis Tema dan Emosi" → "Analisis Emosi" (tema tidak lagi menjadi fokus utama).
- Cakupan data berubah: 15 lagu populer → 54 lagu Hindia di Spotify (sebagai main artist, termasuk kolaborasi atas nama Hindia).
- Evaluasi berubah: deskriptif kualitatif → kuantitatif (accuracy, precision, recall, F1, confusion matrix).
- Dataset training: fine-tuning menggunakan dataset emosi berlabel Indonesia dari Nusacrowd.
- Pengumpulan data: scraping manual dari LyricFind (ada izin langsung dari musisi Hindia untuk keperluan akademik non-komersial).

## Kontribusi yang Diklaim

- Klasifikasi emosi lirik Hindia yang sistematis, terukur, dan dapat dievaluasi secara kuantitatif.
- Pendekatan supervised learning berbasis Transformer multibahasa untuk teks lirik Indonesia.
- Analisis distribusi dan pola emosi antar album serta analisis confidence score.
- Penyajian hasil analisis dalam antarmuka interaktif Streamlit.
