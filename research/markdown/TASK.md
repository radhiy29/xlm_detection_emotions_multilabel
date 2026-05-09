# TASK

## 1. Selaraskan model emosi dengan proposal
- Uraikan bahwa `analysis/emotion_model.py` harus memakai XLM-RoBERTa seperti disebut di Bab III-3.4.6. Jika perlu, jelaskan alasan deviasi beserta dampaknya.
- Tandai perubahan kode dan dokumentasi yang diperlukan agar pipeline emosi benar-benar multibahasa.

## 2. Sesuaikan embedding topik
- Catat bahwa `run_topic_model` saat ini memakai `paraphrase-multilingual-MiniLM-L12-v2`, tetapi proposal menekankan XLM-RoBERTa pada Bab III-3.4.5.
- Tentukan langkah untuk mengganti encoder tersebut (checkpoint, konfigurasi, dan pengujian).

## 3. Kaji cakupan dataset
- Temukan perbedaan antara klaim 15 lagu manual (Bab III-3.2.1) dan file `data/dataset-lirik-lagu-hindia.csv` yang berisi 54 lagu serta uploader generic.
- Putuskan apakah perlu menambah validasi untuk 15 lagu atau memperbarui markdown agar mencerminkan cakupan data terbaru.
