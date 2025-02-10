import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DAELoss(nn.Module):
    def __init__(
            self,
            pad_idx: int,
            label_smoothing: float = 0.1,
            end_weight: float = 3.0,
            char_weight: float = 0.2,
            length_penalty: float = 0.3
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.end_weight = end_weight
        self.char_weight = char_weight
        self.length_penalty = length_penalty

        # Pre-compute difficulty multipliers
        self.difficulty_multipliers = {
            'easy': 1.0,
            'medium': 1.2,
            'hard': 1.5
        }

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction='none'
        )

        # Cache for position weights
        self._position_weights_cache = {}

    @staticmethod
    def _compute_sequence_lengths(tensor: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Optimized sequence length computation"""
        return (tensor != pad_idx).sum(dim=1)

    def get_position_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Optimized position weights calculation"""
        batch_size, seq_len = target.size()
        cache_key = (seq_len, target.device)

        # Check cache first
        if cache_key in self._position_weights_cache:
            weights = self._position_weights_cache[cache_key].clone()
        else:
            # Initialize weights matrix for all possible sequence lengths
            weights = torch.ones(seq_len, seq_len, device=target.device)

            # Pre-compute position importance for all possible lengths
            positions = torch.arange(seq_len, device=target.device).float()
            for length in range(1, seq_len + 1):
                valid_pos = positions[:length] / length
                weights[length - 1, :length] = 1.0 + valid_pos * 0.5

                # End-character weighting
                if length >= 1:
                    weights[length - 1, length - 1] = self.end_weight * 1.5
                if length >= 2:
                    weights[length - 1, length - 2] = self.end_weight * 1.0
                if length >= 3:
                    weights[length - 1, length - 3] = self.end_weight * 0.8

                # Middle characters weighting
                if length >= 4:
                    mid_start = length // 3
                    mid_end = (2 * length) // 3
                    weights[length - 1, mid_start:mid_end] *= 1.3

                # Short word weighting
                if length <= 4:
                    weights[length - 1, :length] *= 1.2

            self._position_weights_cache[cache_key] = weights

        # Get sequence lengths once
        sequence_lengths = self._compute_sequence_lengths(target, self.pad_idx)

        # Use vectorized indexing for faster weight assignment
        batch_weights = torch.ones_like(target, dtype=torch.float)
        for i in range(batch_size):
            length = sequence_lengths[i]
            if length > 0:
                batch_weights[i, :length] = weights[length - 1, :length]

        return batch_weights

    def calculate_length_penalty(self, output_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Optimized length penalty calculation"""
        length_diff = torch.abs(output_lengths.float() - target_lengths.float())

        # Compute all masks at once
        truncation_mask = (output_lengths < target_lengths).float()
        short_prediction_mask = (output_lengths <= 3).float()

        # Vectorized penalty calculation
        total_penalty = length_diff * (1.0 +
                                       truncation_mask * 0.5 +
                                       short_prediction_mask * 0.3)

        return self.length_penalty * total_penalty.mean()

    def calculate_char_ngram_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Optimized n-gram loss calculation"""
        batch_size, seq_len, vocab_size = output.size()
        pred_indices = output.argmax(dim=-1)

        # Pre-compute one-hot encodings
        pred_one_hot = F.one_hot(pred_indices, num_classes=vocab_size)
        target_one_hot = F.one_hot(target, num_classes=vocab_size)

        # Compute bigrams using tensor operations
        pred_bigrams = pred_one_hot[:, :-1] * pred_one_hot[:, 1:]
        target_bigrams = target_one_hot[:, :-1] * target_one_hot[:, 1:]

        # Pre-compute position weights
        bigram_weights = torch.ones(seq_len - 1, device=output.device)
        bigram_weights[-2:] = 1.5

        # Vectorized bigram loss calculation
        bigram_loss = F.mse_loss(
            pred_bigrams.float() * bigram_weights.unsqueeze(0).unsqueeze(-1),
            target_bigrams.float() * bigram_weights.unsqueeze(0).unsqueeze(-1)
        )

        # Only compute trigrams if necessary
        mask = (target != self.pad_idx)
        valid_trigrams = (mask[:, :-2].sum(dim=1) > 0)

        if valid_trigrams.any():
            # Compute trigrams using the pre-computed one-hot encodings
            pred_trigrams = pred_bigrams[:, :-1] * pred_one_hot[:, 2:]
            target_trigrams = target_bigrams[:, :-1] * target_one_hot[:, 2:]

            trigram_weights = torch.ones(seq_len - 2, device=output.device)
            trigram_weights[-2:] = 2.0

            trigram_loss = F.mse_loss(
                pred_trigrams.float() * trigram_weights.unsqueeze(0).unsqueeze(-1),
                target_trigrams.float() * trigram_weights.unsqueeze(0).unsqueeze(-1)
            )

            return bigram_loss + 1.5 * trigram_loss

        return bigram_loss

    def forward(self, output: torch.Tensor, target: torch.Tensor, difficulty: str = 'easy') -> Tuple[
        torch.Tensor, Dict[str, float]]:
        """Optimized forward pass"""
        batch_size, seq_len, vocab_size = output.size()

        # Get sequence lengths once
        padding_mask = (target != self.pad_idx)
        sequence_lengths = self._compute_sequence_lengths(target, self.pad_idx)
        pred_lengths = self._compute_sequence_lengths(output.argmax(dim=-1), self.pad_idx)

        # Compute character prediction loss
        char_loss = self.criterion(
            output.view(-1, vocab_size),
            target.view(-1)
        ).view(batch_size, seq_len)

        # Apply position weights
        weights = self.get_position_weights(target)
        weighted_loss = (char_loss * weights).sum() / weights.sum()

        # Compute penalties
        length_penalty = self.calculate_length_penalty(pred_lengths, sequence_lengths)

        char_ngram_loss = self.calculate_char_ngram_loss(output, target)

        # Apply difficulty multiplier
        multiplier = self.difficulty_multipliers.get(difficulty, 1.0)
        total_loss = multiplier * (
                weighted_loss * 0.7 +
                length_penalty * 0.2 +
                self.char_weight * char_ngram_loss * 0.1
        )

        # Compute metrics efficiently
        with torch.no_grad():
            predictions = output.argmax(dim=-1)
            correct_chars = (predictions == target).float() * padding_mask.float()
            char_accuracy = correct_chars.sum() / padding_mask.sum()

            # Vectorized end character accuracy
            end_indices = sequence_lengths - 1
            valid_ends = end_indices >= 0
            end_char_correct = torch.zeros_like(end_indices, dtype=torch.float)
            end_char_correct[valid_ends] = (
                    predictions[valid_ends, end_indices[valid_ends]] ==
                    target[valid_ends, end_indices[valid_ends]]
            ).float()
            end_char_accuracy = end_char_correct.mean()

            # Length accuracy
            length_match = (sequence_lengths == pred_lengths).float().mean()

        return total_loss, {
            'loss': total_loss.item(),
            'char_prediction_loss': weighted_loss.item(),
            'length_penalty': length_penalty.item(),
            'char_ngram_loss': char_ngram_loss.item(),
            'char_accuracy': char_accuracy.item(),
            'end_char_accuracy': end_char_accuracy.item(),
            'length_accuracy': length_match.item(),
        }