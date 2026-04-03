"""Data loading for machine translation tasks."""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    """Simple translation dataset."""

    def __init__(
        self,
        src_sentences: list[list[int]],
        tgt_sentences: list[list[int]],
        max_len: int = 512,
    ):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src = self.src_sentences[idx][: self.max_len]
        tgt = self.tgt_sentences[idx][: self.max_len]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_idx: int = 0) -> dict[str, torch.Tensor]:
    """Collate function for variable-length sequences."""
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


class Vocabulary:
    """Simple vocabulary class."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str) -> list[int]:
        return [self.sos_idx] + [self.word2idx.get(w, self.unk_idx) for w in sentence.split()] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        words = [self.idx2word.get(i, "<unk>") for i in indices]
        # Remove special tokens
        words = [w for w in words if w not in ["<pad>", "<sos>", "<eos>"]]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    max_len: int = 512,
) -> tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """Get train and validation dataloaders.

    Note: This is a placeholder. For actual training, you would load
    real translation data (e.g., WMT, Multi30k, IWSLT).
    """
    # Example: Create dummy data for testing
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    # Add some example sentences to build vocab
    examples = [
        ("hello world", "hallo welt"),
        ("how are you", "wie geht es dir"),
        ("good morning", "guten morgen"),
        ("thank you very much", "vielen dank"),
        ("see you later", "bis spater"),
    ]

    src_sentences = []
    tgt_sentences = []

    for src, tgt in examples:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
        src_sentences.append(src_vocab.encode(src))
        tgt_sentences.append(tgt_vocab.encode(tgt))

    # Repeat to create a larger dataset
    src_sentences = src_sentences * 100
    tgt_sentences = tgt_sentences * 100

    # Split into train and val
    n_train = int(len(src_sentences) * 0.9)

    train_dataset = TranslationDataset(src_sentences[:n_train], tgt_sentences[:n_train], max_len)
    val_dataset = TranslationDataset(src_sentences[n_train:], tgt_sentences[n_train:], max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
