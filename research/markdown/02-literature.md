# Bab II - Tinjauan Pustaka (Breakdown)

> **Catatan perubahan**: Judul bab sekarang "II. TINJAUAN PUSTAKA" (tanpa prefix "BAB"). Subbab 2.3 berubah dari "Analisis Tema dan Emosi" → "Analisis Emosi" saja. Konten setiap subbab jauh lebih detail dan akademis dengan referensi/sitasi. Tidak ada sintesis BERTopic di bagian tinjauan pustaka — tema hanya sebagai metode pendukung di metodologi.

## 2.1 Musisi Hindia

- Menjelaskan profil Daniel Baskara Putra (Hindia): gaya penulisan reflektif dan personal.
- Album: *Meniari dengan Bayangan* (29 Nov 2019, Sun Eater, 12 lagu + 3 rekaman suara).
- Lagu "Secukupnya" — soundtrack film *Nanti Kita Cerita Tentang Hari Ini* (Jan 2020): >26 juta views YouTube, 71,9 juta pendengar Spotify hingga 2021.
- Lagu "Rumah ke Rumah": narasi emosional personal dengan jutaan pendengar digital.
- Karakteristik lirik Hindia: reflektif, naratif, sarat muatan emosional → layak dianalisis secara komputasional untuk mengidentifikasi dan mengklasifikasikan emosi.

## 2.2 Musik dan Lirik Lagu

- Musik sebagai medium pesan sosial dan ekspresi artistik (etimologi: Yunani *mousikos*).
- Elemen musik: ritme, melodi, harmoni, ekspresi (Jamalus).
- Lirik lagu diperlakukan seperti puisi — kesamaan struktur, rima, gaya bahasa, makna mendalam.
- Lirik mengandung makna semantik dan emosi (eksplisit/implisit).
- Keberhasilan NLP konvensional pada penelitian sebelumnya menjadi landasan pengembangan model bahasa kontekstual Transformer untuk klasifikasi emosi.

## 2.3 Analisis Emosi

- Analisis emosi: mengidentifikasi representasi perasaan dalam teks melalui struktur dan pilihan bahasa.
- Tidak hanya makna harfiah, tetapi aspek simbolik dan kontekstual (metafora, ironi, diksi).
- Emosi disampaikan secara eksplisit maupun implisit dalam lirik yang puitis dan reflektif.
- Contoh: lagu "Bukti" karya Virgoun — bahasa sederhana namun puitis merepresentasikan cinta dan kepedihan secara universal.
- Analisis emosi penting untuk memahami bagaimana teks lirik merepresentasikan perasaan secara sistematis.

> ⚠️ **Perubahan dari versi lama**: subbab ini sebelumnya bernama "Analisis Tema dan Emosi". Sekarang hanya "Analisis Emosi" — analisis tema (BERTopic) dikelola sebagai metode pendukung di metodologi, bukan menjadi fokus utama tinjauan pustaka.

## 2.4 NLP (Natural Language Processing)

- NLP sebagai fondasi untuk memahami dan memproses bahasa manusia secara otomatis.
- Tugas: klasifikasi teks, analisis sentimen dan emosi, ekstraksi informasi.
- Penting: teks harus direpresentasikan ke bentuk numerik sebelum diproses komputer.
- Pada konteks lirik, NLP memungkinkan penggalian ekspresi emosional yang implisit dalam skala besar.
- NLP sebagai fondasi yang menjembatani data teks ke metode pembelajaran mesin.

## 2.5 Machine Learning

- ML: sistem belajar pola dari data secara otomatis tanpa pemrograman eksplisit.
- Data training → melatih model mengenali pola; data testing → evaluasi kemampuan generalisasi.
- Paradigma: supervised learning, unsupervised learning, semi-supervised learning.
- Penelitian ini menggunakan supervised learning karena model dilatih dengan data berlabel emosi dan dievaluasi secara kuantitatif.

## 2.6 Deep Learning

- DL: ML berbasis jaringan saraf tiruan (ANN) berlapis untuk menangkap pola abstrak kompleks.
- Arsitektur umum: CNN (citra), RNN/LSTM (data berurutan).
- Keterbatasan RNN: sulit menangkap konteks jangka panjang pada teks kompleks.
- Solusi: arsitektur Transformer dengan mekanisme attention — memahami konteks tanpa bergantung urutan linear.
- Preprocessing tetap diperlukan sebelum model Transformer diterapkan, agar representasi optimal.

## 2.7 Pre-processing

- Pre-processing: mempersiapkan data teks agar dapat diproses model NLP secara konsisten.
- Tantangan teks: tidak terstruktur, multitafsir, variasi gaya, slang.
- Tahapan umum: cleaning, case folding, normalisasi, tokenisasi, stopword removal, stemming.
- Pada model Transformer: stopword removal dan stemming tidak selalu memberikan dampak positif.
  - Kata seperti "tidak", "namun", "tetapi" penting untuk konteks dan emosi lirik.
  - Stemming berpotensi menghilangkan variasi bentuk kata bermakna.
- Tokenisasi tidak dilakukan manual — diserahkan ke tokenizer internal XLM-RoBERTa.
- Penjelasan ini mendasari strategi preprocessing operasional di Bab III.

## 2.8 BERT (Bidirectional Encoder Representations from Transformers)

- BERT: model representasi teks berbasis Transformer dengan attention bidirectional.
- Kemampuan membaca konteks dua arah → lebih tepat memahami makna kata dalam kalimat.
- Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP).
- Fine-tuning: dapat disesuaikan untuk tugas spesifik tanpa melatih dari awal.
- Menjadi standar baru untuk representasi bahasa alami — melampaui banyak arsitektur tradisional.
- Peran dalam penelitian: landasan konseptual pemodelan bahasa kontekstual → diarahkan ke XLM-RoBERTa untuk klasifikasi emosi.

## 2.9 XLM-RoBERTa

- Pengembangan arsitektur RoBERTa ke lingkungan multibahasa.
- Dilatih pada korpus teks >100 bahasa, termasuk bahasa Indonesia.
- Menangani variasi tanda baca, pilihan kata, dan perbedaan struktur bahasa tanpa preprocessing kompleks.
- Efektif untuk klasifikasi teks bermuatan emosional tinggi (sentimen, emosi).
- Relevan untuk lirik campuran Indonesia-Inggris (code-mixed) yang bersifat reflektif dan metaforis.
- Dipilih sebagai model utama dalam penelitian untuk fine-tuning klasifikasi emosi berbasis supervised learning.

## 2.10 Streamlit

- Framework open-source Python untuk aplikasi web interaktif tanpa penguasaan frontend mendalam.
- Fitur: grafik, tabel, elemen interaktif, serta deploy sebagai aplikasi web yang dapat diakses daring.
- Dalam penelitian: mengintegrasikan klasifikasi emosi berbasis XLM-RoBERTa dengan antarmuka pengguna.
- Menampilkan: hasil evaluasi model + distribusi emosi 54 lirik lagu Hindia secara visual dan interaktif.
- Memperkuat kontribusi Sistem Informasi: bukan hanya model, tetapi juga penyajian hasil analisis.

## 2.11 Penelitian Terdahulu

- Data disajikan dalam bentuk tabel (Tabel 1. Penelitian Terdahulu).
- Tiga studi relevan dikutip:
  1. BERT untuk analisis sentimen lirik bahasa Inggris → akurasi 87% (prediksi popularitas, tanpa klasifikasi emosi spesifik).
  2. Word2Vec + BERT untuk rekomendasi musik → akurasi 84% (evaluasi kesamaan konten, tidak analisis emosi lirik).
  3. Fine-tuning BERT pada ulasan film → accuracy/precision/recall/F1 >0.8 (domain film, bukan lirik).
- Kesenjangan: belum ada studi yang khusus menerapkan Transformer multibahasa (XLM-RoBERTa) untuk **klasifikasi emosi lirik Indonesia** dengan evaluasi kuantitatif komprehensif.

## Sintesis Bab II

- Bab II membangun alur argumentasi: objek lirik Hindia → dasar NLP/ML/DL → model Transformer → pemilihan XLM-RoBERTa → kebutuhan visualisasi Streamlit.
- Titik fokus: analisis emosi (bukan analisis tema) sebagai kontribusi utama.
- Hasil sintesis menjadi landasan desain metodologi di Bab III.
