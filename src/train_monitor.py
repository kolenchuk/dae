import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance


class TrainingMonitor:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.epoch_metrics = defaultdict(list)
        self.current_epoch = defaultdict(float)
        self.gradient_history = defaultdict(list)
        self.attention_patterns = []
        self.num_batches = 0

    def log_batch_metrics(self, loss_dict, predictions, targets, phase='train'):
        pred_texts = [self.dataset.indices_to_text(p.cpu().tolist()) for p in predictions]
        target_texts = [self.dataset.indices_to_text(t.cpu().tolist()) for t in targets]

        char_acc = sum(sum(p == t for p, t in zip(pred, target))
                       for pred, target in zip(pred_texts, target_texts)) / sum(len(t) for t in target_texts)

        lev_distances = [levenshtein_distance(pred, target)
                         for pred, target in zip(pred_texts, target_texts)]
        avg_lev = sum(lev_distances) / len(lev_distances)

        exact_matches = sum(p == t for p, t in zip(pred_texts, target_texts))

        # Accumulate batch metrics
        for metric_name, value in loss_dict.items():
            self.current_epoch[f'{phase}_{metric_name}'] = value

        self.current_epoch[f'{phase}_accuracy'] += char_acc
        self.current_epoch[f'{phase}_levenshtein'] += avg_lev
        self.current_epoch[f'{phase}_exact_matches'] += exact_matches

        error_types = self.analyze_errors(pred_texts, target_texts)
        for error_type, count in error_types.items():
            self.current_epoch[f'{phase}_{error_type}'] += count

        self.num_batches += 1

    def log_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.gradient_history[f'{name}_grad_norm'].append(
                    param.grad.norm().item()
                )

    def analyze_errors(self, predictions, targets):
        error_counts = defaultdict(int)
        for pred, target in zip(predictions, targets):
            if len(pred) != len(target):
                error_counts['length_errors'] += 1
        return error_counts

    def epoch_end(self):
        if self.num_batches == 0:
            return

        for key in self.current_epoch:
            avg_value = self.current_epoch[key] / self.num_batches
            self.epoch_metrics[key].append(avg_value)

        self.current_epoch.clear()
        self.num_batches = 0

    def plot_all_metrics(self, save_path=None):
        plt.figure(figsize=(12, 8))
        metrics_to_plot = [
            ('loss', ['train_loss', 'val_loss']),
            ('accuracy', ['train_accuracy', 'val_accuracy']),
            ('levenshtein', ['train_levenshtein', 'val_levenshtein']),
            ('exact_matches', ['train_exact_matches', 'val_exact_matches'])
        ]

        for metric, keys in metrics_to_plot:
            for key in keys:
                if key in self.epoch_metrics and len(self.epoch_metrics[key]) > 0:
                    plt.plot(self.epoch_metrics[key],
                             label=key.replace('_', ' ').title())

        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_metrics(self, save_path, metrics):
        plt.figure(figsize=(12, 8))

        for key in metrics:
            if key in self.epoch_metrics and len(self.epoch_metrics[key]) > 0:
                plt.plot(self.epoch_metrics[key],
                         label=key.replace('_', ' ').title())

        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()

    def get_latest_metrics(self):
        latest = {}
        for key, values in self.epoch_metrics.items():
            if values:
                latest[key] = values[-1]
        return latest

    def analyze_attention(self, save_path=None):
        if not self.attention_patterns:
            return

        avg_attention = torch.stack(self.attention_patterns).mean(0)

        plt.figure(figsize=(10, 10))
        plt.imshow(avg_attention[0].numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Average Attention Pattern')

        if save_path:
            plt.savefig(save_path)
        plt.close()
