import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F


class DAELoss(nn.Module):
    def __init__(
            self,
            pad_idx: int,
            label_smoothing: float = 0.1,
            end_weight: float = 3.0,
            char_weight: float = 0.2,
            length_penalty: float = 0.1
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.end_weight = end_weight
        self.char_weight = char_weight
        self.length_penalty = length_penalty

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction='none'
        )

    def get_position_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Create weights tensor with enhanced end-character importance
        and position-based weighting.
        """
        weights = torch.ones_like(target, dtype=torch.float)
        padding_mask = (target != self.pad_idx)
        sequence_lengths = padding_mask.sum(dim=1)

        for i in range(target.size(0)):
            length = sequence_lengths[i]
            if length > 0:
                valid_positions = torch.arange(length, device=target.device) / length
                weights[i, :length] = 1.0 + valid_positions * 0.5

                if length >= 1:
                    weights[i, length - 1] = self.end_weight
                if length >= 2:
                    weights[i, length - 2] = self.end_weight * 0.8
                if length >= 3:
                    weights[i, length - 3] = self.end_weight * 0.6

        return weights

    def calculate_length_penalty(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate penalty for length differences between prediction and target.
        """
        pred_lengths = (output.argmax(dim=-1) != self.pad_idx).sum(dim=1)
        target_lengths = (target != self.pad_idx).sum(dim=1)
        length_diff = torch.abs(pred_lengths.float() - target_lengths.float())
        return self.length_penalty * length_diff.mean()

    def calculate_char_ngram_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate character n-gram based loss to capture local patterns.
        """
        batch_size, seq_len, vocab_size = output.size()

        pred_indices = output.argmax(dim=-1)

        pred_bigrams = F.one_hot(pred_indices[:, :-1], num_classes=vocab_size) * \
                       F.one_hot(pred_indices[:, 1:], num_classes=vocab_size)
        target_bigrams = F.one_hot(target[:, :-1], num_classes=vocab_size) * \
                         F.one_hot(target[:, 1:], num_classes=vocab_size)

        mask = (target != self.pad_idx)
        valid_trigrams = (mask[:, :-2].sum(dim=1) > 0)
        trigram_loss = torch.tensor(0.0, device=output.device)

        if valid_trigrams.any():
            pred_trigrams = pred_bigrams[:, :-1] * F.one_hot(pred_indices[:, 2:], num_classes=vocab_size)
            target_trigrams = target_bigrams[:, :-1] * F.one_hot(target[:, 2:], num_classes=vocab_size)
            trigram_loss = F.mse_loss(pred_trigrams.float(), target_trigrams.float())

        return F.mse_loss(pred_bigrams.float(), target_bigrams.float()) + trigram_loss

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Enhanced forward pass with multiple loss components.

        Args:
            output: Model output logits (batch_size, seq_len, vocab_size)
            target: Target indices (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = output.size()

        char_loss = self.criterion(
            output.view(-1, vocab_size),
            target.view(-1)
        ).view(batch_size, seq_len)

        weights = self.get_position_weights(target)
        weighted_loss = (char_loss * weights).sum() / weights.sum()

        length_penalty = self.calculate_length_penalty(output, target)
        char_ngram_loss = self.calculate_char_ngram_loss(output, target)

        total_loss = weighted_loss + length_penalty + self.char_weight * char_ngram_loss

        with torch.no_grad():
            predictions = output.argmax(dim=-1)
            correct_chars = (predictions == target).float() * (target != self.pad_idx).float()

            total_chars = (target != self.pad_idx).sum().float()
            char_accuracy = correct_chars.sum() / total_chars if total_chars > 0 else torch.tensor(0.0)

            padding_mask = (target != self.pad_idx)
            sequence_lengths = padding_mask.sum(dim=1)
            end_char_correct = sum(
                predictions[i, length - 1] == target[i, length - 1]
                for i, length in enumerate(sequence_lengths)
                if length > 0
            )
            end_char_accuracy = end_char_correct / len(sequence_lengths)

            length_match = (sequence_lengths == (predictions != self.pad_idx).sum(dim=1)).float().mean()

        return total_loss, {
            'loss': total_loss.item(),
            'char_prediction_loss': weighted_loss.item(),
            'length_penalty': length_penalty.item(),
            'char_ngram_loss': char_ngram_loss.item(),
            'char_accuracy': char_accuracy.item(),
            'end_char_accuracy': end_char_accuracy,
            'length_accuracy': length_match.item(),
            'combined_loss': total_loss.item()
        }