import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from src.dae_loss import DAELoss
from text_dataset import TextDataset
from train_monitor import TrainingMonitor
from warmup_cosine_scheduler import WarmupCosineScheduler
from collections import defaultdict
from typing import Dict, Tuple, Any
from model_save_utils import save_checkpoint_distributed


class DAETrainer:
    def __init__(self, model: nn.Module, dataset: TextDataset, config: Dict[str, Any]):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = next(model.parameters()).device
        self.best_val_loss = float('inf')

        self.criterion = DAELoss(
            pad_idx=dataset.char_to_idx['<PAD>'],
            end_weight=2.0
        )

        if hasattr(model, 'get_layer_wise_learning_rates'):
            param_groups = model.get_layer_wise_learning_rates(config['learning_rate'])
        else:
            param_groups = model.parameters()

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config['learning_rate'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=config['optimizer']['weight_decay']
        )

        total_steps = config['max_epochs'] * (len(dataset) // config['batch_size'])
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config['lr_schedule']['warmup_steps'],
            total_steps=total_steps,
            min_lr=1e-6
        )

        self.monitor = TrainingMonitor(model, dataset)

        self.scaler = torch.amp.GradScaler(device='cuda',enabled=config['use_amp'])

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_stats = defaultdict(float)
        num_batches = 0

        for batch_idx, (noisy, clean) in enumerate(train_loader):
            try:
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.config['use_amp']):
                    output = self.model(noisy)

                    loss, loss_components = self.criterion(output, clean)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                predictions = output.argmax(dim=-1)
                self.monitor.log_batch_metrics(
                    loss_components,
                    predictions,
                    clean,
                    phase='train'
                )
                self.monitor.log_gradients()

                for key, value in loss_components.items():
                    epoch_stats[key] += value
                num_batches += 1

                if batch_idx % self.config['val_interval'] == 0:
                    self._adjust_training_parameters(loss_components)

            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        self.monitor.epoch_end()
        return {k: v / num_batches for k, v in epoch_stats.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_stats = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                output = self.model(noisy)

                loss, loss_components = self.criterion(output, clean)
                predictions = output.argmax(dim=-1)
                self.monitor.log_batch_metrics(
                    loss_components,
                    predictions,
                    clean,
                    phase='val'
                )

                for key, value in loss_components.items():
                    val_stats[key] += value
                num_batches += 1
        self.monitor.epoch_end()
        metrics = self.monitor.get_latest_metrics()

        if self.config.get('plot_interval'):
            self.monitor.plot_all_metrics(
                os.path.join(self.config['save_dir'], 'training_progress.png')
            )

        return metrics

    def save_checkpoint(self, save_dir: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)

        if (metrics['combined_loss'] < self.best_val_loss):
            self.best_val_loss = metrics['combined_loss']
            best_checkpoint_path = os.path.join(save_dir, 'dae_best_checkpoint')
            save_checkpoint_distributed(
                self.model,
                best_checkpoint_path,
                self.dataset.char_to_idx,
                self.dataset.idx_to_char,
                {
                    'epoch': epoch,
                    'metrics': metrics,
                    'vocab_size': self.dataset.vocab_size,
                }
            )

        if self.monitor:
            self.monitor.plot_all_metrics(os.path.join(save_dir, 'training_metrics.png'))
            self.monitor.analyze_attention(os.path.join(save_dir, 'attention_patterns.png'))

    def plot_metrics(self, save_dir: str):
        """Plot training metrics."""
        metrics_to_plot = [
            ('loss', ['train_loss', 'val_loss']),
            ('accuracy', ['train_accuracy', 'val_accuracy']),
            ('levenshtein', ['train_levenshtein', 'val_levenshtein']),
            ('exact_matches', ['train_exact_matches', 'val_exact_matches'])
        ]
        for metric, keys in metrics_to_plot:
            self.monitor.plot_metrics(os.path.join(save_dir, f'{metric}_plot.png'), keys)


    def _adjust_training_parameters(self, loss_components: Dict[str, float]):
        """Dynamically adjust training parameters based on loss components."""
        if loss_components['end_char_accuracy'] < 0.7:
            self.criterion.end_weight = min(5.0, self.criterion.end_weight * 1.1)
        elif loss_components['end_char_accuracy'] > 0.9:
            self.criterion.end_weight = max(1.0, self.criterion.end_weight * 0.9)

        if loss_components['char_ngram_loss'] > 0.5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.95

        if loss_components['char_accuracy'] < 0.8:
            self.criterion.char_weight = min(0.5, self.criterion.char_weight * 1.1)
