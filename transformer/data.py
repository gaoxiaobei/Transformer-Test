"""Data loading for machine translation tasks."""

import os
import urllib.request
import zipfile
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    """Vocabulary for tokenization."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.strip().split():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str) -> list[int]:
        return [self.sos_idx] + [self.word2idx.get(w, self.unk_idx) for w in sentence.strip().split()] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        words = [self.idx2word.get(i, "<unk>") for i in indices]
        words = [w for w in words if w not in ["<pad>", "<sos>", "<eos>"]]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


class TranslationDataset(Dataset):
    """Translation dataset."""

    def __init__(self, src_sentences: list[str], tgt_sentences: list[str], src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int = 512):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src = self.src_vocab.encode(self.src_sentences[idx])[: self.max_len]
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx])[: self.max_len]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_idx: int = 0) -> dict[str, torch.Tensor]:
    """Pad sequences in batch."""
    src_lens = [item["src"].size(0) for item in batch]
    tgt_lens = [item["tgt"].size(0) for item in batch]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    src_batch = torch.zeros(len(batch), max_src_len, dtype=torch.long).fill_(pad_idx)
    tgt_batch = torch.zeros(len(batch), max_tgt_len, dtype=torch.long).fill_(pad_idx)

    for i, item in enumerate(batch):
        src_batch[i, : item["src"].size(0)] = item["src"]
        tgt_batch[i, : item["tgt"].size(0)] = item["tgt"]

    return {"src": src_batch, "tgt": tgt_batch}


def download_multi30k(data_dir: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Download and load Multi30k dataset (de-en)."""
    os.makedirs(data_dir, exist_ok=True)

    # Multi30k URLs
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = {
        "train.de": "train.de.gz",
        "train.en": "train.en.gz",
        "val.de": "val.de.gz",
        "val.en": "val.en.gz",
    }

    import gzip

    data = {}
    for name, remote in files.items():
        local_path = os.path.join(data_dir, name)
        gz_path = os.path.join(data_dir, remote)

        if not os.path.exists(local_path):
            print(f"Downloading {remote}...")
            urllib.request.urlretrieve(base_url + remote, gz_path)
            with gzip.open(gz_path, "rb") as f_in:
                with open(local_path, "wb") as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)

        with open(local_path, "r", encoding="utf-8") as f:
            data[name] = [line.strip() for line in f.readlines()]

    return data["train.de"], data["train.en"], data["val.de"], data["val.en"]


def get_dataloaders(
    data_dir: str = "./transformer/data",
    batch_size: int = 64,
    num_workers: int = 4,
    max_len: int = 128,
    vocab_min_freq: int = 2,
) -> tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """Get train and validation dataloaders for machine translation (de-en)."""

    # Download/load data
    train_de, train_en, val_de, val_en = download_multi30k(data_dir)

    print(f"Train samples: {len(train_de)}")
    print(f"Val samples: {len(val_de)}")

    # Build vocabularies
    from collections import Counter

    def build_vocab(sentences: list[str], min_freq: int = 2) -> Vocabulary:
        vocab = Vocabulary()
        word_counts = Counter()
        for sent in sentences:
            word_counts.update(sent.strip().split())

        for word, count in word_counts.items():
            if count >= min_freq:
                vocab.add_sentence(word)

        return vocab

    src_vocab = build_vocab(train_de, vocab_min_freq)
    tgt_vocab = build_vocab(train_en, vocab_min_freq)

    print(f"Source vocab (de): {len(src_vocab)} words")
    print(f"Target vocab (en): {len(tgt_vocab)} words")

    # Create datasets
    train_dataset = TranslationDataset(train_de, train_en, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_de, val_en, src_vocab, tgt_vocab, max_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
        pin_memory=True,
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
