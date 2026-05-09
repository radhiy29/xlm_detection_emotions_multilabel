import math
import streamlit as st
from transformers import pipeline

# ── Label emosi (5 kelas) ─────────────────────────────────────────────────────
LABELS = ["Senang", "Sedih", "Marah", "Takut", "Cinta"]

# Path model hasil fine-tuning
MODEL_PATH = "models/finetuned-xlm-multilabel-twostage"

# Label mapping: mendukung label English (fallback) dan Indonesian (lokal)
EMOTION_MAP = {
    # English labels (fallback model)
    "joy":      "Senang",
    "happy":    "Senang",
    "sadness":  "Sedih",
    "anger":    "Marah",
    "fear":     "Takut",
    "love":     "Cinta",
    "optimism": "Senang",
    "disgust":  "Marah",
    "neutral":  "Sedih",
    # Indonesian labels (model lokal fine-tuned)
    "senang":   "Senang",
    "sedih":    "Sedih",
    "marah":    "Marah",
    "takut":    "Takut",
    "cinta":    "Cinta",
}

@st.cache_resource(show_spinner="Memuat Model Emosi Baru...")
def get_emotion_model_v3():
    """
    Load model klasifikasi emosi XLM-RoBERTa sesuai pilihan.
    Di-cache oleh Streamlit agar model tidak dimuat ulang pada setiap rerun.
    """
    classifier = pipeline(
        task="text-classification",
        model=MODEL_PATH,
        return_all_scores=True,
    )
    classifier._loaded_model_id = MODEL_PATH
    print(f"[emotion_model] Model loaded: {MODEL_PATH}")
    return classifier


def predict_emotion(model, text: str):
    """
    Inferensi emosi dengan normalisasi label.
    Selalu mengembalikan (label_indonesia, skor).
    Mendukung model lokal (label Indonesia) dan fallback (label English).
    """
    # Biarkan tokenizer yang mengurus pemotongan token (truncation)
    result = model(text, truncation=True, max_length=128)

    # Normalisasi format output pipeline
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            scores = result
        elif isinstance(result[0], list):
            scores = result[0]
        else:
            raise ValueError(f"Format output model tidak dikenali: {type(result[0])}")
    else:
        raise ValueError(f"Output model kosong atau tidak valid: {result}")

    # Temperature Scaling untuk melembutkan distribusi probabilitas
    # Semakin tinggi T, semakin rata probabilitasnya. T=5.0 akan membuat max ~0.6
    TEMPERATURE = 5.0
    
    # Hitung logit semu dari probabilitas, lalu terapkan temperature
    pseudo_logits = [math.log(max(item["score"], 1e-9)) / TEMPERATURE for item in scores]
    exp_logits = [math.exp(l) for l in pseudo_logits]
    sum_exp = sum(exp_logits)
    
    # Update scores dengan probabilitas yang sudah dilembutkan
    for i, item in enumerate(scores):
        item["score"] = exp_logits[i] / sum_exp

    # Filter label yang dikenali
    filtered = [
        item for item in scores
        if item["label"].lower() in EMOTION_MAP
    ]

    if filtered:
        top = max(filtered, key=lambda x: x["score"])
        label_key = top["label"].lower()
        return EMOTION_MAP[label_key], round(top["score"], 4)

    # Fallback absolut
    top_all = max(scores, key=lambda x: x["score"])
    label_lower = top_all["label"].lower()
    if label_lower in EMOTION_MAP:
        return EMOTION_MAP[label_lower], round(top_all["score"], 4)

    return "Tidak Diketahui", round(top_all["score"], 4)
