import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")


def tokenized_fn(example):
    output = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    return output


def datapipe(dataset):
    dataset = dataset.map(tokenized_fn, batched=True)
    dataset = dataset.remove_columns(["id", "text"])
    return dataset


data = load_dataset(
    "json",
    data_files={
        "train": "./data/nlpcc2024/track1_train.jsonl",
        "test": "./data/nlpcc2024/track1_val.jsonl",
    },
    num_proc=8,
)


data = datapipe(data)
data.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=5,
    problem_type="multi_label_classification",
)

model.requires_grad = False
model.classifier.requires_grad=True

training_argus = TrainingArguments(
    output_dir="./model/",
    eval_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_train_epochs=5,
    save_strategy="epoch",
    remove_unused_columns=True,
    load_best_model_at_end=True,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    labels = labels.astype(int).reshape(-1)
    score = f1_score(labels, predictions, average="macro")
    return {"f1": score}


trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    compute_metrics=compute_metrics,
    args=training_argus,
)


trainer.train()
