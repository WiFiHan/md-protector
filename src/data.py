"""Loaders for MAGE, M4 and RAID. Each returns {split: [(text, label), ...]}, label 0 = machine, 1 = human."""
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_mage(path, splits):
    """MAGE: one CSV per split. The unseen domain/model settings also have test_ood.csv."""
    data = {}
    for split in splits:
        df = pd.read_csv(os.path.join(path, f"{split}.csv"), usecols=["text", "label", "src"])
        data[split] = [(text, int(label)) for text, label in zip(df["text"].tolist(), df["label"].tolist())]
    return data


def load_m4(path, splits):
    """M4 (SemEval-2024 Task 8, Subtask A). Split names are like "multilingual_train"."""
    data = {}
    for split in splits:
        track, part = split.split("_", 1)
        records = []
        with open(os.path.join(path, f"subtaskA_{part}_{track}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.decoder.JSONDecodeError:
                    continue
                # M4 uses 1 for machine text, so flip the label
                records.append((row["text"], 1 if row["label"] == 0 else 0))
        data[split] = records
    return data


def load_raid(manifest_path, splits):
    """RAID (Shengkun/Raid_split). train90/valid10 are taken from its train split using the
    index file in splits/raid/."""
    from datasets import load_dataset

    data = {}
    if any(split in ("train90", "valid10") for split in splits):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        raid_train = load_dataset("Shengkun/Raid_split", split="train")
        if int(manifest["source_train_size"]) != len(raid_train):
            raise ValueError(f"RAID manifest expects {manifest['source_train_size']} train rows, found {len(raid_train)}")
        train_indices = set(int(i) for i in manifest["train_indices"])
        valid_indices = set(int(i) for i in manifest["valid_indices"])
        train90, valid10 = [], []
        for i, (text, model) in enumerate(zip(raid_train["generation"], raid_train["model"])):
            record = (text, 1 if model == "human" else 0)
            if i in train_indices:
                train90.append(record)
            elif i in valid_indices:
                valid10.append(record)
            else:
                raise ValueError(f"RAID train row {i} is not covered by the split manifest")
        data["train90"], data["valid10"] = train90, valid10
    if "test" in splits:
        raid_test = load_dataset("Shengkun/Raid_split", split="test")
        data["test"] = [(text, 1 if model == "human" else 0)
                        for text, model in zip(raid_test["generation"], raid_test["model"])]
    missing = [s for s in splits if s not in data]
    if missing:
        raise KeyError(f"Unknown RAID split(s): {missing}")
    return data


def load_splits(opt, splits):
    if opt.dataset == "mage":
        return load_mage(opt.data_path, splits)
    if opt.dataset == "m4":
        return load_m4(opt.data_path, splits)
    if opt.dataset == "raid":
        return load_raid(opt.raid_split_manifest, splits)
    raise ValueError(f"Unknown dataset: {opt.dataset}")


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch, tokenizer, max_length=512):
    texts = [text for text, _ in batch]
    labels = torch.tensor([label for _, label in batch])
    encoded = tokenizer(texts, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    return encoded, labels
