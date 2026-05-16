import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.text_cleaner import clean_lyrics


MULTILABEL_LABELS = ["Senang", "Sedih", "Marah", "Takut", "Cinta"]
LABEL2ID = {label: i for i, label in enumerate(MULTILABEL_LABELS)}


def labels_to_multihot(label_text: str) -> list[int]:
    vec = np.zeros(len(MULTILABEL_LABELS), dtype=int)
    for label in str(label_text).split(","):
        label = label.strip()
        if label in LABEL2ID:
            vec[LABEL2ID[label]] = 1
    return vec.tolist()


@st.cache_data(show_spinner="Memuat data uji multilabel lirik...")
def load_lyrics_multilabel_splits(path="dataset/dataset-emosi-multilabel.csv", random_state=42):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        text = clean_lyrics(str(row.get("lirik_lagu", "")))
        labels = labels_to_multihot(row.get("label_emosi", ""))
        if text and sum(labels) > 0:
            rows.append({"text": text, "labels": labels})
    data = pd.DataFrame(rows)
    label_totals = np.array(data["labels"].tolist()).sum(axis=0)

    def split_key(vec):
        active = np.where(np.array(vec) > 0)[0]
        rarest = active[np.argmin(label_totals[active])]
        return MULTILABEL_LABELS[int(rarest)]

    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=random_state, stratify=data["labels"].apply(split_key))
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["labels"].apply(split_key))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


FINAL_MODEL_PATH = "models/finetuned-xlm-multilabel-twostage"


@st.cache_resource(show_spinner="Memuat model multilabel...")
def load_multilabel_model(model_path=FINAL_MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    metadata_path = os.path.join(model_path, "multilabel_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    thresholds = metadata.get("thresholds", {})
    threshold_values = np.array([thresholds.get(label, 0.5) for label in MULTILABEL_LABELS], dtype=np.float32)
    return tokenizer, model, device, threshold_values, metadata


def predict_multilabel_texts(tokenizer, model, device, texts, thresholds, max_length=192, batch_size=16):
    probs_all = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.vstack(probs_all) if probs_all else np.empty((0, len(MULTILABEL_LABELS)))
    preds = (probs >= thresholds).astype(int)
    empty = preds.sum(axis=1) == 0
    if empty.any():
        preds[empty, probs[empty].argmax(axis=1)] = 1
    return probs, preds


def split_lyrics_to_stanzas(lyrics: str, max_words=120) -> list[str]:
    text = str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[.*?\]", "\n\n", text)
    raw_stanzas = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not raw_stanzas:
        raw_stanzas = [line.strip() for line in text.split("\n") if line.strip()]

    stanzas = []
    for stanza in raw_stanzas:
        cleaned = clean_lyrics(stanza)
        words = cleaned.split()
        if not words:
            continue
        if len(words) <= max_words:
            stanzas.append(cleaned)
        else:
            for start in range(0, len(words), max_words):
                chunk = " ".join(words[start:start + max_words]).strip()
                if chunk:
                    stanzas.append(chunk)
    return stanzas


def aggregate_stanza_scores(stanza_probs, mean_weight=0.7):
    probs = np.array(stanza_probs, dtype=np.float32)
    if probs.ndim == 1:
        return probs
    mean_scores = probs.mean(axis=0)
    max_scores = probs.max(axis=0)
    return (mean_weight * mean_scores) + ((1 - mean_weight) * max_scores)


def multilabel_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_label = pd.DataFrame({
        "Label": MULTILABEL_LABELS,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Jumlah Sampel": support,
    })
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_label": per_label,
        "confusion_matrices": multilabel_confusion_matrix(y_true, y_pred),
    }


def labels_from_vector(vec) -> str:
    labels = [label for label, active in zip(MULTILABEL_LABELS, vec) if int(active) == 1]
    return ", ".join(labels) if labels else "Tidak Diketahui"


def dominant_label_from_probs(probs) -> str:
    return MULTILABEL_LABELS[int(np.argmax(probs))]


def cooccurrence_matrix(label_vectors):
    matrix = np.array(label_vectors, dtype=int)
    return matrix.T @ matrix
