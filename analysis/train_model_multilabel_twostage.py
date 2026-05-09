import argparse
import json
import os
import re
from itertools import product

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


LABELS = ["Senang", "Sedih", "Marah", "Takut", "Cinta"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

TWEET_URLS = {
    "anger": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/AngerData.csv",
    "fear": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/FearData.csv",
    "joy": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/JoyData.csv",
    "love": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/LoveData.csv",
    "sad": "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion/SadData.csv",
}

TWEET_MAP = {
    "joy": "Senang",
    "sad": "Sedih",
    "anger": "Marah",
    "fear": "Takut",
    "love": "Cinta",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def labels_to_multihot(label_text: str) -> list[float]:
    labels = [label.strip() for label in str(label_text).split(",")]
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for label in labels:
        if label in LABEL2ID:
            vec[LABEL2ID[label]] = 1.0
    return vec.tolist()


def load_hindia_multilabel(path="dataset/dataset-emosi-multilabel.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        text = clean_text(str(row.get("lirik_lagu", "")))
        labels = labels_to_multihot(row.get("label_emosi", ""))
        if text and sum(labels) > 0:
            rows.append({"text": text, "labels": labels, "source": "hindia"})
    return pd.DataFrame(rows)


def split_hindia(df: pd.DataFrame, random_state=42):
    label_totals = np.array(df["labels"].tolist()).sum(axis=0)

    def split_key(vec):
        active = np.where(np.array(vec) > 0)[0]
        if len(active) == 0:
            return "none"
        rarest = active[np.argmin(label_totals[active])]
        return LABELS[int(rarest)]

    keys = df["labels"].apply(split_key)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=keys
    )
    temp_keys = temp_df["labels"].apply(split_key)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_keys
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_tweet_multihot(tweet_per_label=200, random_state=42) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_state)
    for source_label, url in TWEET_URLS.items():
        target_label = TWEET_MAP[source_label]
        df = pd.read_csv(url, sep="\t")
        text_col = "Tweet" if "Tweet" in df.columns else df.columns[-1]
        texts = []
        for text in df[text_col].dropna().astype(str):
            cleaned = clean_text(text)
            word_count = len(cleaned.split())
            if 3 <= word_count <= 60:
                texts.append(cleaned)
        texts = list(dict.fromkeys(texts))
        sample_size = min(tweet_per_label, len(texts))
        if sample_size < len(texts):
            idx = rng.choice(len(texts), size=sample_size, replace=False)
            texts = [texts[i] for i in idx]
        vec = np.zeros(len(LABELS), dtype=np.float32)
        vec[LABEL2ID[target_label]] = 1.0
        for text in texts:
            rows.append({"text": text, "labels": vec.tolist(), "source": "tweet"})
    return pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)


def cap_pos_weight(labels_matrix, min_weight=1.0, max_weight=4.0):
    positives = labels_matrix.sum(axis=0)
    negatives = labels_matrix.shape[0] - positives
    weights = negatives / np.maximum(positives, 1)
    return np.clip(weights, min_weight, max_weight).astype(np.float32)


def compute_multilabel_metrics(predictions, label_ids, thresholds=None):
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    probs = 1 / (1 + np.exp(-logits))
    if thresholds is None:
        thresholds = np.full(len(LABELS), 0.5)
    preds = (probs >= thresholds).astype(int)
    empty = preds.sum(axis=1) == 0
    if empty.any():
        preds[empty, probs[empty].argmax(axis=1)] = 1
    y_true = label_ids.astype(int)
    return {
        "f1_micro": f1_score(y_true, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, preds, average="macro", zero_division=0),
    }


def default_thresholds_metrics(logits, y_true):
    return compute_multilabel_metrics(logits, y_true, np.full(len(LABELS), 0.5))


def tune_thresholds(logits, y_true):
    probs = 1 / (1 + np.exp(-logits))
    thresholds = []
    per_label = {}
    for i, label in enumerate(LABELS):
        best_t, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.2, 0.76, 0.05):
            pred = (probs[:, i] >= threshold).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_t, best_f1 = float(threshold), float(score)
        thresholds.append(best_t)
        per_label[label] = {"threshold": best_t, "f1": best_f1}
    return np.array(thresholds, dtype=np.float32), per_label


class MultilabelTrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        pos_weight = self.pos_weight.to(outputs.logits.device) if self.pos_weight is not None else None
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(df, tokenizer, max_length):
    dataset = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize_fn, batched=True)


def train_once(args, tweet_per_label, stage2_lr, stage2_max_length=None, stage2_weight_decay=None):
    stage2_max_length = stage2_max_length or args.stage2_max_length
    stage2_weight_decay = stage2_weight_decay if stage2_weight_decay is not None else args.stage2_weight_decay
    os.makedirs(args.output_root, exist_ok=True)
    run_name = f"tweet{tweet_per_label}_lr{stage2_lr:g}_len{stage2_max_length}_wd{stage2_weight_decay:g}".replace(".", "p")
    stage1_dir = os.path.join(args.output_root, f"stage1_{run_name}")
    final_dir = os.path.join(args.output_root, f"final_{run_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA aktif: {device_name} ({total_vram:.1f} GB VRAM)")
    else:
        print("CUDA tidak tersedia. Training berjalan di CPU dan akan jauh lebih lambat.")
    hindia_df = load_hindia_multilabel(args.lyrics_path)
    train_h, val_h, test_h = split_hindia(hindia_df, args.random_state)
    tweet_df = load_tweet_multihot(tweet_per_label, args.random_state)

    print("=" * 70)
    print(f"Run: {run_name}")
    print(f"Hindia split -> train={len(train_h)} val={len(val_h)} test={len(test_h)}")
    print(f"Tweet stage-1 -> {len(tweet_df)} rows ({tweet_per_label}/label target)")

    tweet_train_ds = tokenize_dataset(tweet_df, tokenizer, args.stage1_max_length)
    val_ds_stage1 = tokenize_dataset(val_h, tokenizer, args.stage1_max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    stage1_args = TrainingArguments(
        output_dir=stage1_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.stage1_lr,
        per_device_train_batch_size=args.stage1_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.stage1_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        logging_steps=20,
        fp16=args.fp16 and torch.cuda.is_available(),
        tf32=args.tf32 and torch.cuda.is_available(),
        gradient_accumulation_steps=args.stage1_gradient_accumulation,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    stage1_trainer = MultilabelTrainer(
        model=model,
        args=stage1_args,
        train_dataset=tweet_train_ds,
        eval_dataset=val_ds_stage1,
        data_collator=collator,
        compute_metrics=lambda p: compute_multilabel_metrics(p.predictions, p.label_ids),
    )
    print("Stage 1: fine-tuning on balanced tweet subset...")
    stage1_trainer.train()
    stage1_trainer.save_model(stage1_dir)
    tokenizer.save_pretrained(stage1_dir)

    labels_matrix = np.array(train_h["labels"].tolist(), dtype=np.float32)
    pos_weight = cap_pos_weight(labels_matrix)
    print("Stage 2 pos_weight:", dict(zip(LABELS, pos_weight.round(3).tolist())))

    train_h_stage2 = train_h.copy()
    if args.oversample_rare:
        rare_rows = train_h_stage2[
            train_h_stage2["labels"].apply(lambda v: v[LABEL2ID["Takut"]] == 1 or v[LABEL2ID["Senang"]] == 1)
        ]
        train_h_stage2 = pd.concat([train_h_stage2, rare_rows], ignore_index=True).sample(frac=1, random_state=args.random_state)

    train_ds = tokenize_dataset(train_h_stage2, tokenizer, stage2_max_length)
    val_ds = tokenize_dataset(val_h, tokenizer, stage2_max_length)
    test_ds = tokenize_dataset(test_h, tokenizer, stage2_max_length)

    model = stage1_trainer.model

    stage2_args = TrainingArguments(
        output_dir=final_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=stage2_lr,
        per_device_train_batch_size=args.stage2_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.stage2_epochs,
        weight_decay=stage2_weight_decay,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        logging_steps=10,
        fp16=args.fp16 and torch.cuda.is_available(),
        tf32=args.tf32 and torch.cuda.is_available(),
        gradient_accumulation_steps=args.stage2_gradient_accumulation,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    stage2_trainer = MultilabelTrainer(
        pos_weight=pos_weight.tolist(),
        model=model,
        args=stage2_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=lambda p: compute_multilabel_metrics(p.predictions, p.label_ids),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    print("Stage 2: fine-tuning on Hindia multilabel lyrics...")
    stage2_trainer.train()

    val_pred = stage2_trainer.predict(val_ds)
    thresholds, threshold_report = tune_thresholds(val_pred.predictions, val_pred.label_ids.astype(int))
    val_metrics_default = default_thresholds_metrics(val_pred.predictions, val_pred.label_ids.astype(int))
    val_metrics_tuned = compute_multilabel_metrics(val_pred.predictions, val_pred.label_ids.astype(int), thresholds)
    test_pred = stage2_trainer.predict(test_ds)
    test_metrics = compute_multilabel_metrics(test_pred.predictions, test_pred.label_ids, thresholds)

    final_export_dir = args.final_output_dir if args.single_run else final_dir
    os.makedirs(final_export_dir, exist_ok=True)
    stage2_trainer.save_model(final_export_dir)
    tokenizer.save_pretrained(final_export_dir)

    metadata = {
        "model_type": "multilabel_twostage",
        "labels": LABELS,
        "tweet_per_label": tweet_per_label,
        "stage1_lr": args.stage1_lr,
        "stage2_lr": stage2_lr,
        "stage2_max_length": stage2_max_length,
        "stage2_weight_decay": stage2_weight_decay,
        "thresholds": dict(zip(LABELS, thresholds.round(4).tolist())),
        "threshold_report": threshold_report,
        "pos_weight": dict(zip(LABELS, pos_weight.round(4).tolist())),
        "validation_metrics_default_threshold": {k: round(float(v), 4) for k, v in val_metrics_default.items()},
        "validation_metrics_tuned_threshold": {k: round(float(v), 4) for k, v in val_metrics_tuned.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "split_sizes": {"train": len(train_h), "validation": len(val_h), "test": len(test_h)},
    }
    with open(os.path.join(final_export_dir, "multilabel_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Test metrics:", metadata["test_metrics"])
    print("Thresholds:", metadata["thresholds"])
    return {"run_name": run_name, "output_dir": final_export_dir, **metadata}


def main():
    parser = argparse.ArgumentParser(description="Two-stage multilabel XLM-RoBERTa fine-tuning for Hindia lyrics")
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--lyrics_path", default="dataset/dataset-emosi-multilabel.csv")
    parser.add_argument("--output_root", default="models/experiments-xlm-multilabel-twostage")
    parser.add_argument("--final_output_dir", default="models/finetuned-xlm-multilabel-twostage")
    parser.add_argument("--tweet_per_label", type=int, default=200)
    parser.add_argument("--tweet_sizes", default="100,200,300")
    parser.add_argument("--stage1_lr", type=float, default=2e-5)
    parser.add_argument("--stage2_lrs", default="1e-5,5e-6")
    parser.add_argument("--stage2_max_lengths", default="192")
    parser.add_argument("--stage2_weight_decays", default="0.05")
    parser.add_argument("--stage1_epochs", type=float, default=2)
    parser.add_argument("--stage2_epochs", type=float, default=8)
    parser.add_argument("--stage1_batch_size", type=int, default=16)
    parser.add_argument("--stage2_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--stage1_gradient_accumulation", type=int, default=1)
    parser.add_argument("--stage2_gradient_accumulation", type=int, default=1)
    parser.add_argument("--stage1_max_length", type=int, default=128)
    parser.add_argument("--stage2_max_length", type=int, default=192)
    parser.add_argument("--stage2_weight_decay", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--grid", action="store_true", help="Run tweet-size and LR grid search")
    parser.add_argument("--oversample_rare", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    args = parser.parse_args()

    if args.grid:
        args.single_run = False
        tweet_sizes = [int(x.strip()) for x in args.tweet_sizes.split(",") if x.strip()]
        stage2_lrs = [float(x.strip()) for x in args.stage2_lrs.split(",") if x.strip()]
        stage2_max_lengths = [int(x.strip()) for x in args.stage2_max_lengths.split(",") if x.strip()]
        stage2_weight_decays = [float(x.strip()) for x in args.stage2_weight_decays.split(",") if x.strip()]
        results = [
            train_once(args, t, lr, max_len, wd)
            for t, lr, max_len, wd in product(tweet_sizes, stage2_lrs, stage2_max_lengths, stage2_weight_decays)
        ]
        best = max(results, key=lambda r: r["validation_metrics_tuned_threshold"].get("f1_macro", 0))
        print("=" * 70)
        print("Best run by validation macro F1:", best["run_name"], best["validation_metrics_tuned_threshold"])
        print("Held-out test metrics:", best["test_metrics"])
        with open(os.path.join(args.output_root, "grid_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        args.single_run = True
        train_once(args, args.tweet_per_label, 1e-5)


if __name__ == "__main__":
    main()
