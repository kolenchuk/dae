import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


def convert_tensor_to_python(obj: Any) -> Any:
    """Convert PyTorch tensors to Python native types recursively."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_python(item) for item in obj)
    return obj


class CurriculumMetricsAnalyzer:
    def __init__(self, metrics: Dict, save_dir: str):
        """
        Initialize the analyzer with curriculum training metrics.
        """
        self.metrics = self._convert_tensor_to_python(metrics)
        self.save_dir = save_dir
        self.reports_dir = os.path.join(save_dir, 'train_reports')
        self.plots_dir = os.path.join(save_dir, 'plots')

        # Create directories if they don't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _convert_tensor_to_python(self, obj: Any) -> Any:
        """Convert PyTorch tensors to Python native types recursively."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensor_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensor_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_tensor_to_python(item) for item in obj)
        return obj

    def save_metrics_report(self):
        """Save metrics to JSON file"""
        report_path = os.path.join(self.reports_dir, 'curriculum_metrics.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            print(f"Metrics saved to {report_path}")
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")

    def extract_metric_values(self, phase_metrics: List[Dict], metric_key: str) -> Tuple[List[float], List[float]]:
        """
        Extract train and validation values for a specific metric with improved error handling
        and metric key detection.
        """
        train_values = []
        val_values = []

        for epoch_data in phase_metrics:
            train_metrics = epoch_data.get('train_metrics', {})
            val_metrics = epoch_data.get('val_metrics', {})

            # Try different possible metric key formats
            possible_train_keys = [
                metric_key,
                f'train_{metric_key}',
                metric_key.replace('train_', '')
            ]
            possible_val_keys = [
                metric_key,
                f'val_{metric_key}',
                metric_key.replace('val_', '')
            ]

            # Find the first available train metric key
            train_value = None
            for key in possible_train_keys:
                if key in train_metrics:
                    value = train_metrics[key]
                    if isinstance(value, (int, float, torch.Tensor)):
                        train_value = float(value) if isinstance(value, (int, float)) else float(value[0])
                    break

            # Find the first available validation metric key
            val_value = None
            for key in possible_val_keys:
                if key in val_metrics:
                    value = val_metrics[key]
                    if isinstance(value, (int, float, torch.Tensor)):
                        val_value = float(value) if isinstance(value, (int, float)) else float(value[0])
                    break

            if train_value is not None:
                train_values.append(train_value)
            if val_value is not None:
                val_values.append(val_value)

        return train_values, val_values

    def plot_phase_comparison(self, metric_name: str):
        """Create comparison plot for a specific metric across all phases with improved error handling"""
        plt.figure(figsize=(12, 6))

        phases = ['phase1_metrics', 'phase2_metrics', 'phase3_metrics']
        phase_names = ['Easy', 'Medium', 'Hard']
        colors = ['blue', 'green', 'red']

        has_data = False
        for phase, phase_name, color in zip(phases, phase_names, colors):
            if phase in self.metrics:
                train_values, val_values = self.extract_metric_values(self.metrics[phase], metric_name)

                if train_values and val_values:  # Only plot if we have data
                    has_data = True
                    epochs = range(1, len(train_values) + 1)

                    plt.plot(epochs, train_values, color, label=f'{phase_name} Train', alpha=0.7)
                    plt.plot(epochs, val_values, linestyle='--', color=color, label=f'{phase_name} Val', alpha=0.7)

        if has_data:
            plt.title(f'{metric_name.replace("_", " ").title()} Across Training Phases')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
            plt.legend()

            plot_path = os.path.join(self.plots_dir, f'{metric_name}_phase_comparison.png')
            plt.savefig(plot_path)

        plt.close()

    def plot_all_metrics(self):
        """Generate plots for all metrics"""
        metrics_to_plot = [
            'loss',
            'accuracy',
            'levenshtein',
            'exact_matches',
            'char_accuracy',
            'end_char_accuracy',
            'length_accuracy',
            'char_ngram_loss'
        ]

        for metric in metrics_to_plot:
            try:
                self.plot_phase_comparison(metric)
            except Exception as e:
                print(f"Error plotting {metric}: {str(e)}")

    def get_best_metric(self, metrics_list, key, operation='max', default=0):
        """Helper function to get best metric value while handling invalid values"""
        valid_values = []
        for metrics in metrics_list:
            value = convert_tensor_to_python(metrics['val_metrics'].get(key, default))
            # Skip infinity, NaN, and None values
            if isinstance(value, (int, float)) and value != float('inf') and value != float('-inf') and not np.isnan(
                    value):
                valid_values.append(value)

        if not valid_values:
            return default

        return max(valid_values) if operation == 'max' else min(valid_values)

    def generate_summary_report(self):
        """Generate a summary of the best metrics for each phase with improved metric extraction"""
        summary = {}

        for phase in ['phase1_metrics', 'phase2_metrics', 'phase3_metrics']:
            if phase in self.metrics:
                phase_data = self.metrics[phase]

                # Initialize with default values
                best_metrics = {
                    'best_accuracy': 0.0,
                    'best_char_accuracy': 0.0,
                    'lowest_loss': float('inf'),
                    'lowest_levenshtein': float('inf'),
                    'total_epochs': len(phase_data)
                }

                # Extract metrics for each epoch
                for epoch_data in phase_data:
                    val_metrics = epoch_data.get('val_metrics', {})

                    # Update best metrics
                    for metric_name, current_value in val_metrics.items():
                        try:
                            value = float(current_value) if isinstance(current_value, (int, float)) else float(
                                current_value[0])

                            if 'accuracy' in metric_name:
                                best_metrics['best_accuracy'] = max(best_metrics['best_accuracy'], value)
                            if 'char_accuracy' in metric_name:
                                best_metrics['best_char_accuracy'] = max(best_metrics['best_char_accuracy'], value)
                            if 'loss' in metric_name:
                                best_metrics['lowest_loss'] = min(best_metrics['lowest_loss'], value)
                            if 'levenshtein' in metric_name:
                                best_metrics['lowest_levenshtein'] = min(best_metrics['lowest_levenshtein'], value)
                        except (TypeError, ValueError) as e:
                            print(f"Error processing metric {metric_name}: {str(e)}")
                            continue

                summary[phase] = best_metrics

        # Save summary report
        summary_path = os.path.join(self.reports_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        return summary


def analyze_curriculum_metrics(metrics: Dict, save_dir: str) -> Dict:
    """
    Main function to analyze and visualize curriculum training metrics.
    """
    analyzer = CurriculumMetricsAnalyzer(metrics, save_dir)

    # Save raw metrics
    with open(os.path.join(analyzer.reports_dir, 'curriculum_metrics.json'), 'w') as f:
        json.dump(analyzer.metrics, f, indent=4)

    # Generate plots
    analyzer.plot_all_metrics()

    # Generate and save summary
    summary = analyzer.generate_summary_report()

    return summary