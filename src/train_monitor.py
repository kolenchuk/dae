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
            self.current_epoch[metric_name] += value

        self.current_epoch['accuracy'] += char_acc
        self.current_epoch['levenshtein'] += avg_lev
        self.current_epoch['exact_matches'] += exact_matches

        error_types = self.analyze_errors(pred_texts, target_texts)
        for error_type, count in error_types.items():
            self.current_epoch[error_type] += count

        self.num_batches += 1

    def analyze_errors(self, predictions, targets):
        error_counts = defaultdict(int)
        for pred, target in zip(predictions, targets):
            if len(pred) != len(target):
                error_counts['length_errors'] += 1
        return error_counts

    def epoch_end(self):
        epoch_metrics = defaultdict(float)

        if self.num_batches == 0:
            return epoch_metrics

        for key in self.current_epoch:
            epoch_metrics[key] = self.current_epoch[key] / self.num_batches

        self.current_epoch.clear()
        self.num_batches = 0

        return epoch_metrics