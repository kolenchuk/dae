import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from typing import List, Tuple
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_text(text):
    """ Приводить текст до нижнього регістру та очищає від зайвих пробілів. """
    if not isinstance(text, str):
        text = str(text) if pd.notnull(text) else ""
    return text.lower().strip()


def pad_collate_fn(batch):
    """Padding sequences to the same length in a batch"""
    noisy, clean = zip(*batch)

    noisy = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in noisy]
    clean = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in clean]

    max_len = max(max(len(seq) for seq in noisy), max(len(seq) for seq in clean))

    noisy_padded = torch.stack([
        F.pad(seq, (0, max_len - len(seq)), value=0, mode='constant')
        for seq in noisy
    ])

    clean_padded = torch.stack([
        F.pad(seq, (0, max_len - len(seq)), value=0, mode='constant')
        for seq in clean
    ])

    return noisy_padded, clean_padded


class TextDataset(Dataset):
    def __init__(self, csv_file: str, max_length: int = 100):
        """
        Initialize dataset.

        Args:
            csv_file: Path to CSV file with noisy and clean text
            max_length: Maximum sequence length
        """
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length

        self.build_vocabulary()

        self.clean_indices = [
            self.text_to_indices(text.lower().strip(), add_eos=True)
            for text in self.data['clean_output']
        ]
        self.noisy_indices = [
            self.text_to_indices(text.lower().strip(), add_eos=False)
            for text in self.data['noisy_input']
        ]

    def build_vocabulary(self):
        """Build character vocabulary from dataset."""
        all_text = (
                "".join(self.data['noisy_input'].astype(str)) +
                "".join(self.data['clean_output'].astype(str))
        )
        char_counts = Counter(all_text.lower())

        self.char_to_idx = {}

        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<EOS>'] = 1

        idx = 2
        for char, _ in sorted(char_counts.items()):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                idx += 1

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        self.special_tokens = {
            self.char_to_idx['<PAD>'],
            self.char_to_idx['<EOS>']
        }

        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
        logger.info(f"Number of unique characters: {len(char_counts)}")

    def text_to_indices(self, text: str, add_eos: bool = False) -> List[int]:
        """
        Convert text to list of indices.

        Args:
            text: Input text
            add_eos: Whether to add EOS token

        Returns:
            List of indices
        """
        indices = []

        for char in text:
            if char.lower() in self.char_to_idx:
                indices.append(self.char_to_idx[char.lower()])
            else:
                logger.warning(f"Unknown character encountered: {char}")
                continue

        if add_eos:
            indices.append(self.char_to_idx['<EOS>'])

        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            if add_eos:
                indices[-1] = self.char_to_idx['<EOS>']

        return indices

    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert indices back to text.

        Args:
            indices: List of indices

        Returns:
            Reconstructed text
        """
        return "".join([
            self.idx_to_char[idx]
            for idx in indices
            if idx not in self.special_tokens and idx in self.idx_to_char
        ])

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (noisy, clean) tensors
        """
        noisy = torch.tensor(self.noisy_indices[idx], dtype=torch.long)
        clean = torch.tensor(self.clean_indices[idx], dtype=torch.long)
        return noisy, clean

    def decode(self, indices: List[int]) -> str:
        """Alias for indices_to_text for compatibility."""
        return self.indices_to_text(indices)