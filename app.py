import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.multilabel_evaluator import (
    MULTILABEL_LABELS,
    aggregate_stanza_scores,
    dominant_label_from_probs,
    labels_from_vector,
    load_lyrics_multilabel_splits,
    load_multilabel_model,
    multilabel_metrics,
    predict_multilabel_texts,
    split_lyrics_to_stanzas,
)
from utils.helpers import validate_dataset


st.set_page_config(page_title="Analisis Lirik Hindia", layout="wide", page_icon="🎵")

EMOTION_COLORS = {
    "Senang": "#DDA15E",
    "Sedih": "#6F8FAF",
    "Marah": "#C97064",
    "Takut": "#8E7AB5",
    "Cinta": "#C88FA3",
}
CHART_COLORS = ["#DDA15E", "#6F8FAF", "#C97064", "#8E7AB5", "#C88FA3", "#7A9E7E"]
HEATMAP_SCALE = [[0, "#F8F5F0"], [0.35, "#E8C7A7"], [0.7, "#C9866A"], [1, "#6F4E37"]]

st.markdown(
    """
<style>
.section-title { font-size: 1.1rem; font-weight: 600; color: #555; margin-bottom: 0.3rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
.app-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.65rem;
    height: 1.65rem;
    margin-right: 0.42rem;
    border-radius: 0.55rem;
    background: #F6EFE8;
    color: #6F4E37;
    font-size: 0.95rem;
    line-height: 1;
    vertical-align: 0.02rem;
}
[data-testid="stMetric"] {
    min-height: 116px;
    padding: 1rem 0.75rem;
    border: 1px solid #E9E2DA;
    border-radius: 16px;
    background: #FBF8F4;
    text-align: center;
}
[data-testid="stMetricLabel"] {
    justify-content: center;
    min-height: 2.4rem;
}
[data-testid="stMetricLabel"] p {
    width: 100%;
    color: #6B625B;
    font-size: 0.88rem;
    font-weight: 600;
    line-height: 1.15;
    text-align: center;
    white-space: normal;
}
[data-testid="stMetricValue"] {
    justify-content: center;
    color: #3F4E4F;
    font-size: 1.85rem;
    font-weight: 700;
}
[data-testid="stDownloadButton"] button {
    min-height: 0;
    padding: 0.34rem 0.78rem;
    border: 1px solid #E9E2DA;
    border-radius: 999px;
    background: #FBF8F4;
    color: #6F4E37;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.15;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #C9866A;
    background: #F6EFE8;
    color: #6F4E37;
}
.download-button-wrap {
    margin-top: 0.65rem;
    margin-bottom: 1.15rem;
}
.centered-table-wrap {
    min-height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.combo-table-wrap {
    min-height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.styled-table-wrap {
    width: 100%;
}
.centered-table-wrap table,
.combo-table-wrap table,
.styled-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    border: 1px solid #E9E2DA;
    border-radius: 14px;
    overflow: hidden;
    background: #FFFFFF;
}
.centered-table-wrap th,
.combo-table-wrap th,
.styled-table-wrap th {
    background: #F6EFE8;
    color: #3F4E4F;
    font-weight: 700;
    text-align: center;
    padding: 0.72rem 0.6rem;
}
.centered-table-wrap td,
.combo-table-wrap td,
.styled-table-wrap td {
    text-align: center;
    padding: 0.68rem 0.6rem;
    border-top: 1px solid #EFE7DF;
}
.preview-table-wrap {
    width: 100%;
    max-height: 520px;
    overflow: auto;
    border: 1px solid #E9E2DA;
    border-radius: 14px;
    background: #FFFFFF;
}
.preview-table-wrap table {
    width: 100%;
    border-collapse: collapse;
}
.preview-table-wrap th {
    position: sticky;
    top: 0;
    background: #F6EFE8;
    color: #3F4E4F;
    font-weight: 700;
    text-align: center;
    padding: 0.72rem 0.6rem;
    z-index: 1;
}
.preview-table-wrap td {
    max-width: 380px;
    padding: 0.68rem 0.6rem;
    border-top: 1px solid #EFE7DF;
    vertical-align: top;
}
.preview-table-wrap td:not(:last-child) {
    text-align: center;
}
.preview-table-wrap td:last-child {
    text-align: left;
    line-height: 1.45;
}
.prediction-table-wrap {
    width: 100%;
    max-height: 560px;
    overflow: auto;
    border: 1px solid #E9E2DA;
    border-radius: 14px;
    background: #FFFFFF;
}
.prediction-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.prediction-table-wrap th {
    position: sticky;
    top: 0;
    background: #F6EFE8;
    color: #3F4E4F;
    font-weight: 700;
    text-align: center;
    padding: 0.72rem 0.6rem;
    z-index: 1;
}
.prediction-table-wrap td {
    padding: 0.68rem 0.6rem;
    border-top: 1px solid #EFE7DF;
    text-align: center;
    vertical-align: top;
}
.prediction-table-wrap td:first-child,
.prediction-table-wrap td:nth-child(2),
.prediction-table-wrap td:nth-child(3) {
    text-align: left;
    min-width: 150px;
}
.prediction-table-wrap td:nth-last-child(-n+5) {
    font-variant-numeric: tabular-nums;
}
.stanza-table-wrap {
    width: 100%;
    max-height: 560px;
    overflow: auto;
    border: 1px solid #E9E2DA;
    border-radius: 14px;
    background: #FFFFFF;
}
.stanza-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.stanza-table-wrap th {
    position: sticky;
    top: 0;
    background: #F6EFE8;
    color: #3F4E4F;
    font-weight: 700;
    text-align: center;
    padding: 0.72rem 0.6rem;
    z-index: 1;
}
.stanza-table-wrap td {
    padding: 0.68rem 0.6rem;
    border-top: 1px solid #EFE7DF;
    text-align: center;
    vertical-align: top;
}
.stanza-table-wrap td:nth-child(2),
.stanza-table-wrap td:nth-child(3),
.stanza-table-wrap td:nth-child(4) {
    text-align: left;
}
.stanza-table-wrap td:nth-child(4) {
    min-width: 300px;
    max-width: 520px;
    line-height: 1.45;
}
.stanza-table-wrap td:nth-last-child(-n+5) {
    font-variant-numeric: tabular-nums;
}
</style>
""",
    unsafe_allow_html=True,
)


def section_title(icon: str, text: str, level=4):
    st.markdown(f'{"#" * level} <span class="app-icon">{icon}</span>{text}', unsafe_allow_html=True)


def download_table(df: pd.DataFrame, file_name: str, key: str):
    st.markdown('<div class="download-button-wrap">', unsafe_allow_html=True)
    st.download_button(
        label="Unduh CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def model_session_key(prefix: str, model_name: str) -> str:
    return f"{prefix}_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"


def metric_bar(metrics: dict):
    rows = [
        {"Metrik": "Micro F1", "Nilai": metrics.get("micro_f1", 0)},
        {"Metrik": "Macro F1", "Nilai": metrics.get("macro_f1", 0)},
        {"Metrik": "Weighted F1", "Nilai": metrics.get("weighted_f1", 0)},
        {"Metrik": "Macro Precision", "Nilai": metrics.get("precision_macro", 0)},
        {"Metrik": "Macro Recall", "Nilai": metrics.get("recall_macro", 0)},
    ]
    fig = px.bar(pd.DataFrame(rows), x="Metrik", y="Nilai", text_auto=".2%", color_discrete_sequence=["#4F6D7A"])
    fig.update_layout(height=360, yaxis_tickformat=".0%", yaxis_range=[0, 1])
    return fig


def chart_note(text: str):
    st.caption(text)


def render_discography_analysis(df_emo: pd.DataFrame):
    pred_matrix = np.array(df_emo["pred_vector"].tolist(), dtype=int)

    combo_df = df_emo["label_prediksi"].value_counts().reset_index()
    combo_df.columns = ["Kombinasi Emosi", "Jumlah"]
    cardinality_df = df_emo["jumlah_emosi_terdeteksi"].value_counts().sort_index().reset_index()
    cardinality_df.columns = ["Jumlah Emosi", "Jumlah Lagu"]
    cardinality_df["Kategori Jumlah Emosi"] = cardinality_df["Jumlah Emosi"].astype(str)
    fig_cardinality = px.bar(
        cardinality_df,
        x="Jumlah Emosi",
        y="Jumlah Lagu",
        text_auto=True,
        color="Kategori Jumlah Emosi",
        color_discrete_sequence=CHART_COLORS + list(EMOTION_COLORS.values()),
    )
    fig_cardinality.update_layout(height=380, showlegend=False)

    cardinality_col, combo_col = st.columns(2, gap="large")
    with cardinality_col:
        section_title("📏", "Jumlah Emosi per Lagu")
        st.metric("Rata-rata jumlah emosi terdeteksi per lagu", f'{df_emo["jumlah_emosi_terdeteksi"].mean():.2f}')
        st.plotly_chart(fig_cardinality, use_container_width=True)
        chart_note("Menunjukkan berapa banyak label emosi yang aktif dalam satu lagu. Bar yang lebih tinggi berarti jumlah lagu dengan total emosi tersebut lebih banyak.")
    with combo_col:
        section_title("🧩", "Kombinasi Emosi Terbanyak")
        st.markdown(
            f'<div class="styled-table-wrap">{combo_df.head(12).to_html(index=False)}</div>',
            unsafe_allow_html=True,
        )
        download_table(combo_df, "kombinasi_emosi_terbanyak.csv", "download_combo_table")

    freq_df = pd.DataFrame({"Emosi": MULTILABEL_LABELS, "Jumlah": pred_matrix.sum(axis=0).astype(int)})
    fig_freq = px.pie(freq_df, names="Emosi", values="Jumlah", color="Emosi", color_discrete_map=EMOTION_COLORS)
    fig_freq.update_layout(height=420)

    if "album" in df_emo.columns:
        album_rows = []
        for album, group in df_emo.groupby("album"):
            counts = np.array(group["pred_vector"].tolist(), dtype=int).sum(axis=0)
            for label, count in zip(MULTILABEL_LABELS, counts):
                album_rows.append({"Album": album, "Emosi": label, "Jumlah": int(count)})
        album_df = pd.DataFrame(album_rows)
        pivot = album_df.pivot(index="Album", columns="Emosi", values="Jumlah").fillna(0)
        fig_album_heat = px.imshow(pivot, text_auto=True, color_continuous_scale=HEATMAP_SCALE)
        fig_album_heat.update_layout(height=420)

        album_heat_col, freq_col = st.columns(2, gap="large")
        with album_heat_col:
            section_title("🗂️", "Heatmap Emosi x Album")
            st.plotly_chart(fig_album_heat, use_container_width=True)
            chart_note("Menampilkan jumlah kemunculan tiap emosi pada setiap album. Warna yang lebih pekat menunjukkan emosi tersebut lebih sering terdeteksi pada album terkait.")
        with freq_col:
            section_title("🥧", "Distribusi Emosi")
            st.plotly_chart(fig_freq, use_container_width=True)
            chart_note("Menunjukkan proporsi seluruh label emosi yang terdeteksi dari semua lagu. Bagian terbesar berarti emosi tersebut paling dominan secara keseluruhan.")
    else:
        section_title("🥧", "Distribusi Emosi")
        st.plotly_chart(fig_freq, use_container_width=True)
        chart_note("Menunjukkan proporsi seluruh label emosi yang terdeteksi dari semua lagu. Bagian terbesar berarti emosi tersebut paling dominan secara keseluruhan.")

    if "album" in df_emo.columns:
        section_title("📊", "Bar Confidence per Album")
        album_selected = st.selectbox("Pilih album:", df_emo["album"].unique().tolist())
        album_scores = df_emo[df_emo["album"] == album_selected].melt(id_vars=["judul_lagu"], value_vars=MULTILABEL_LABELS, var_name="Emosi", value_name="Skor")
        fig_album = px.bar(
            album_scores,
            x="judul_lagu",
            y="Skor",
            color="Emosi",
            barmode="group",
            range_y=[0, 1],
            color_discrete_map=EMOTION_COLORS,
        )
        fig_album.update_layout(height=430, xaxis_tickangle=-30)
        st.plotly_chart(fig_album, use_container_width=True)
        chart_note("Membandingkan skor tiap emosi pada lagu-lagu dalam album yang dipilih. Skor mendekati 1 berarti emosi tersebut lebih kuat menurut model.")

    if "tahun_rilis" in df_emo.columns:
        section_title("📅", "Tren Emosi per Tahun Rilis")
        trend_rows = []
        for year, group in df_emo.groupby("tahun_rilis"):
            counts = np.array(group["pred_vector"].tolist(), dtype=int).sum(axis=0)
            for label, count in zip(MULTILABEL_LABELS, counts):
                trend_rows.append({"Tahun": year, "Emosi": label, "Jumlah": int(count)})
        trend_df = pd.DataFrame(trend_rows)
        fig_trend = px.line(
            trend_df,
            x="Tahun",
            y="Jumlah",
            color="Emosi",
            markers=True,
            color_discrete_map=EMOTION_COLORS,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        chart_note("Menampilkan perubahan jumlah emosi terdeteksi berdasarkan tahun rilis. Garis yang naik menunjukkan emosi tersebut lebih sering muncul pada tahun tersebut.")


selected_model = "XLM-RoBERTa Multilabel"


st.title("Analisis Emosi Multilabel — Lirik Lagu Hindia")
st.caption("XLM-RoBERTa · Two-Stage Fine-Tuning · Streamlit | Sistem Informasi, UNJA")
st.divider()

tabs = st.tabs(["Evaluasi Model", "Input Data", "Analisis Emosi"])


with tabs[0]:
    section_title("📊", "Evaluasi Model Multilabel", level=3)
    st.markdown(
        "Evaluasi dijalankan pada test-set lirik Hindia saja. Model memakai threshold hasil validasi lirik untuk menghasilkan prediksi multilabel."
    )

    eval_key = model_session_key("eval", selected_model)
    if st.button(f"▶ Jalankan Evaluasi ({selected_model})", key="run_eval_multilabel"):
        try:
            tokenizer, model, device, thresholds, _ = load_multilabel_model()
            _, _, test_df = load_lyrics_multilabel_splits()
            probs, preds = predict_multilabel_texts(tokenizer, model, device, test_df["text"].tolist(), thresholds)
            y_true = np.array(test_df["labels"].tolist(), dtype=int)
            res = multilabel_metrics(y_true, preds)
            st.session_state[eval_key] = res
        except Exception as e:
            st.error(f"Evaluasi gagal: {e}")

    eval_result = st.session_state.get(eval_key)
    if eval_result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Micro F1", f'{eval_result.get("micro_f1", 0):.2%}')
        c2.metric("Macro F1", f'{eval_result.get("macro_f1", 0):.2%}')
        c3.metric("Weighted F1", f'{eval_result.get("weighted_f1", 0):.2%}')
        c4.metric("Macro Precision", f'{eval_result.get("precision_macro", 0):.2%}')
        c5.metric("Macro Recall", f'{eval_result.get("recall_macro", 0):.2%}')

        st.plotly_chart(metric_bar(eval_result), use_container_width=True)
        st.markdown(
            """
**Cara Membaca Ringkasan Metrik**
- `Micro F1` dipakai untuk melihat performa model secara keseluruhan pada semua label.
- `Macro F1` membantu melihat apakah performa model relatif merata antar emosi, termasuk label yang jumlah datanya lebih sedikit.
- `Weighted F1` membaca performa dengan mempertimbangkan jumlah data tiap label, sehingga label yang lebih sering muncul memberi pengaruh lebih besar.
- Jika `Macro Precision` lebih tinggi dari `Macro Recall`, model cenderung lebih selektif memberi label.
- Jika `Macro Recall` lebih tinggi dari `Macro Precision`, model cenderung lebih banyak menangkap label, tetapi perlu dilihat apakah false positive meningkat.
"""
        )

        section_title("🎯", "Precision, Recall, F1 per Label")
        per_label = eval_result["per_label"].round(4)
        label_table_col, label_chart_col = st.columns(2, gap="large")
        with label_table_col:
            st.markdown(
                f'<div class="centered-table-wrap">{per_label.to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )
            download_table(per_label, "evaluasi_per_label.csv", "download_eval_per_label")

        fig_label = px.bar(
            per_label,
            x="Label",
            y=["Precision", "Recall", "F1"],
            barmode="group",
            range_y=[0, 1],
            color_discrete_sequence=CHART_COLORS,
        )
        fig_label.update_layout(height=380, yaxis_tickformat=".0%")
        with label_chart_col:
            st.plotly_chart(fig_label, use_container_width=True)
        st.markdown(
            """
**Cara Membaca Performa per Label**
- Bandingkan `F1` antar emosi untuk melihat label mana yang relatif lebih mudah atau lebih sulit diprediksi model.
- `Precision` tinggi pada suatu emosi berarti prediksi untuk emosi tersebut cenderung tepat.
- `Recall` tinggi pada suatu emosi berarti model cukup mampu menemukan data yang memang memiliki emosi tersebut.
- Jika precision tinggi tetapi recall rendah, model cenderung hati-hati dan sebagian label benar mungkin tidak terdeteksi.
- Jika recall tinggi tetapi precision rendah, model cenderung sering memberi label tersebut dan perlu dilihat kemungkinan prediksi berlebih.
"""
        )

        section_title("🧮", "Multilabel Confusion Matrix per Label")
        cm_rows = []
        for label, cm in zip(MULTILABEL_LABELS, eval_result["confusion_matrices"]):
            tn, fp, fn, tp = cm.ravel()
            cm_rows.append({"Label": label, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
        cm_df = pd.DataFrame(cm_rows)
        cm_chart_col, cm_table_col = st.columns(2, gap="large")
        fig_cm = px.imshow(cm_df.set_index("Label")[["TP", "FP", "FN", "TN"]], text_auto=True, color_continuous_scale=HEATMAP_SCALE)
        fig_cm.update_layout(height=380, xaxis_title="Komponen", yaxis_title="Label")
        with cm_chart_col:
            st.plotly_chart(fig_cm, use_container_width=True)
        with cm_table_col:
            st.markdown(
                f'<div class="centered-table-wrap">{cm_df.to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )
            download_table(cm_df, "confusion_matrix_multilabel.csv", "download_confusion_matrix")
        st.markdown(
            """
**Cara Membaca Confusion Matrix**
- `TP` menunjukkan jumlah data yang benar-benar memiliki label emosi tersebut dan berhasil terdeteksi.
- `FP` menunjukkan prediksi berlebih, yaitu model memberi label emosi tersebut padahal label sebenarnya tidak ada.
- `FN` menunjukkan label yang terlewat, yaitu emosi sebenarnya ada tetapi tidak diprediksi oleh model.
- `TN` menunjukkan data yang memang tidak memiliki label tersebut dan model juga tidak memprediksinya.
- Jika `FP` tinggi, model cenderung terlalu mudah memberi label tersebut; jika `FN` tinggi, model cenderung kurang sensitif terhadap label tersebut.
"""
        )

with tabs[1]:
    section_title("📂", "Upload Dataset Lirik", level=3)
    st.markdown("Upload dataset lirik lagu Hindia dalam format CSV. Kolom wajib: `judul_lagu`, `musisi`, `album`, `tahun_rilis`, dan `lirik_lagu`.")
    file = st.file_uploader("Pilih file CSV dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        try:
            validate_dataset(df)
            st.session_state["df"] = df
            st.success(f"Dataset valid. {len(df)} baris siap diproses.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Baris", len(df))
            c2.metric("Total Album", df["album"].nunique() if "album" in df.columns else 0)
            c3.metric("Rata-rata Panjang Lirik", f'{int(df["lirik_lagu"].str.len().mean())} karakter')
            st.markdown("#### Preview Dataset")
            preview_cols = [col for col in ["judul_lagu", "musisi", "album", "tahun_rilis", "lirik_lagu"] if col in df.columns]
            preview_df = df[preview_cols].copy()
            if "lirik_lagu" in preview_df.columns:
                preview_df["lirik_lagu"] = preview_df["lirik_lagu"].astype(str).str.slice(0, 260) + "..."
            st.markdown(
                f'<div class="preview-table-wrap">{preview_df.head(20).to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )
            download_table(preview_df, "preview_dataset_lirik.csv", "download_preview_dataset")
            if len(df) > 20:
                st.caption(f"Menampilkan 20 baris pertama dari total {len(df)} baris.")
        except ValueError as e:
            st.error(str(e))
    else:
        st.info("Belum ada dataset yang diupload.")


df = st.session_state.get("df")


with tabs[2]:
    if df is None:
        st.warning("Upload dataset terlebih dahulu di tab Input Data.")
    else:
        section_title("💬", "Analisis Emosi Multilabel", level=3)
        st.markdown(
            "Lirik utuh dipecah menjadi bait berdasarkan newline kosong, diprediksi per bait, lalu diagregasikan menjadi skor level lagu."
        )
        result_key = model_session_key("emotion_results", selected_model)
        if st.button(f"▶ Jalankan Analisis ({selected_model})", key="run_multilabel"):
            try:
                tokenizer, model, device, thresholds, _ = load_multilabel_model()
                song_scores = []
                song_preds = []
                stanza_details = []
                progress = st.progress(0, text="Menganalisis lirik per bait...")

                for row_idx, row in df.iterrows():
                    stanzas = split_lyrics_to_stanzas(row.get("lirik_lagu", ""))
                    if not stanzas:
                        stanzas = [""]

                    stanza_probs, stanza_preds = predict_multilabel_texts(
                        tokenizer, model, device, stanzas, thresholds
                    )
                    song_prob = aggregate_stanza_scores(stanza_probs)
                    song_pred = (song_prob >= thresholds).astype(int)
                    if song_pred.sum() == 0:
                        song_pred[int(np.argmax(song_prob))] = 1

                    song_scores.append(song_prob)
                    song_preds.append(song_pred)
                    for stanza_no, (stanza_text, probs_row, preds_row) in enumerate(zip(stanzas, stanza_probs, stanza_preds), start=1):
                        detail = {
                            "row_id": row_idx,
                            "judul_lagu": row.get("judul_lagu", f"Lagu {row_idx + 1}"),
                            "bait_ke": stanza_no,
                            "bait": stanza_text,
                            "label_prediksi": labels_from_vector(preds_row),
                            "emosi_dominan": dominant_label_from_probs(probs_row),
                        }
                        for i, label in enumerate(MULTILABEL_LABELS):
                            detail[label] = round(float(probs_row[i]), 4)
                        stanza_details.append(detail)
                    progress.progress((row_idx + 1) / len(df), text=f"Menganalisis lirik per bait... {row_idx + 1}/{len(df)}")

                progress.empty()
                probs = np.array(song_scores, dtype=np.float32)
                preds = np.array(song_preds, dtype=int)
                df_emo = df.copy()
                for i, label in enumerate(MULTILABEL_LABELS):
                    df_emo[label] = probs[:, i].round(4)
                df_emo["label_prediksi"] = [labels_from_vector(row) for row in preds]
                df_emo["emosi_dominan"] = [dominant_label_from_probs(row) for row in probs]
                df_emo["jumlah_emosi_terdeteksi"] = preds.sum(axis=1)
                df_emo["pred_vector"] = preds.tolist()
                df_emo["jumlah_bait"] = [len(split_lyrics_to_stanzas(x)) for x in df["lirik_lagu"].tolist()]
                st.session_state[result_key] = df_emo
                st.session_state[model_session_key("stanza_details", selected_model)] = pd.DataFrame(stanza_details)
                st.session_state[model_session_key("thresholds", selected_model)] = thresholds
                st.success(f"Analisis selesai. {len(df_emo)} lagu diproses per bait.")
            except Exception as e:
                st.error(f"Analisis gagal: {e}")

        df_emo = st.session_state.get(result_key)
        if df_emo is not None:
            section_title("🏷️", "Tabel Prediksi Multilabel")
            base_cols = ["judul_lagu", "label_prediksi", "emosi_dominan", "jumlah_emosi_terdeteksi", "jumlah_bait"]
            if "album" in df_emo.columns:
                base_cols.insert(1, "album")
            prediction_df = df_emo[base_cols + MULTILABEL_LABELS].copy()
            prediction_df = prediction_df.rename(columns={
                "judul_lagu": "Judul Lagu",
                "album": "Album",
                "label_prediksi": "Label Prediksi",
                "emosi_dominan": "Emosi Dominan",
                "jumlah_emosi_terdeteksi": "Jumlah Emosi",
                "jumlah_bait": "Jumlah Bait",
            })
            for label in MULTILABEL_LABELS:
                prediction_df[label] = prediction_df[label].map(lambda value: f"{float(value):.2f}")
            st.markdown(
                f'<div class="prediction-table-wrap">{prediction_df.to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )
            download_table(prediction_df, "prediksi_multilabel_lagu.csv", "download_prediction_table")
            st.caption("Skor tiap emosi berada pada rentang 0-1. Label prediksi ditentukan dari skor yang melewati threshold masing-masing emosi.")

            threshold_values = st.session_state.get(model_session_key("thresholds", selected_model))
            if threshold_values is None:
                _, _, _, threshold_values, _ = load_multilabel_model()
            with st.expander("Aturan Penentuan Label Emosi"):
                st.markdown(
                    """
Threshold ditentukan dari **data validasi lirik Hindia** pada tahap evaluasi model. Nilai threshold dicari per label emosi untuk memilih ambang yang memberi F1 terbaik pada data validasi.

- Setiap label punya threshold sendiri karena distribusi tiap emosi tidak sama.
- Saat inferensi, label dianggap aktif jika skor model lebih besar atau sama dengan threshold label tersebut.
- Threshold ini tidak dilatih sebagai bobot model, tetapi dipakai sebagai aturan keputusan setelah model menghasilkan skor sigmoid.
"""
                )
                threshold_df = pd.DataFrame({"Label": MULTILABEL_LABELS, "Threshold": threshold_values}).round(4)
                st.markdown(
                    f'<div class="styled-table-wrap">{threshold_df.to_html(index=False)}</div>',
                    unsafe_allow_html=True,
                )
                download_table(threshold_df, "threshold_label_emosi.csv", "download_threshold_table")

            lagu = st.selectbox("Pilih lagu:", df_emo["judul_lagu"].tolist(), key="profile_lagu")
            row = df_emo[df_emo["judul_lagu"] == lagu].iloc[0]

            radar_values = [row[label] for label in MULTILABEL_LABELS]
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=MULTILABEL_LABELS + [MULTILABEL_LABELS[0]],
                fill="toself",
                line=dict(color="#8E7AB5", width=3),
                fillcolor="rgba(142, 122, 181, 0.22)",
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=420)

            stanza_key = model_session_key("stanza_details", selected_model)
            stanza_df = st.session_state.get(stanza_key)
            if stanza_df is not None:
                stanza_song = stanza_df[stanza_df["judul_lagu"] == lagu].copy()
                stanza_long = stanza_song.melt(
                    id_vars=["bait_ke"],
                    value_vars=MULTILABEL_LABELS,
                    var_name="Emosi",
                    value_name="Skor",
                )
                fig_timeline = px.line(
                    stanza_long,
                    x="bait_ke",
                    y="Skor",
                    color="Emosi",
                    markers=True,
                    range_y=[0, 1],
                    color_discrete_map=EMOTION_COLORS,
                )
                fig_timeline.update_layout(height=420, xaxis_title="Bait ke-", yaxis_title="Skor")

                radar_col, timeline_col = st.columns(2, gap="large")
                with radar_col:
                    section_title("🕸️", "Peta Intensitas Emosi")
                    st.plotly_chart(fig_radar, use_container_width=True)
                    chart_note("Menunjukkan profil skor emosi untuk lagu yang dipilih. Area yang lebih melebar pada suatu emosi berarti skor emosi tersebut lebih kuat.")
                with timeline_col:
                    section_title("📈", "Tren Skor Emosi per Bait")
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    chart_note("Menampilkan perubahan skor emosi dari bait ke bait. Kenaikan garis menunjukkan emosi tersebut makin kuat pada bait tertentu.")

                section_title("📄", "Rincian Skor Emosi per Bait")
                stanza_detail_df = stanza_song[["bait_ke", "label_prediksi", "emosi_dominan", "bait"] + MULTILABEL_LABELS].copy()
                stanza_detail_df = stanza_detail_df.rename(columns={
                    "bait_ke": "Bait Ke-",
                    "label_prediksi": "Label Prediksi",
                    "emosi_dominan": "Emosi Dominan",
                    "bait": "Teks Bait",
                })
                for label in MULTILABEL_LABELS:
                    stanza_detail_df[label] = stanza_detail_df[label].map(lambda value: f"{float(value):.2f}")
                st.markdown(
                    f'<div class="stanza-table-wrap">{stanza_detail_df.to_html(index=False)}</div>',
                    unsafe_allow_html=True,
                )
                download_table(stanza_detail_df, "rincian_skor_emosi_per_bait.csv", "download_stanza_detail")

            render_discography_analysis(df_emo)
