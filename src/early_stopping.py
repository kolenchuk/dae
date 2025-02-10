from typing import Dict


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.best_composite_score = None
        self.patience = patience
        self.counter = 0
        self.min_epoch = 2
        self.patience_by_phase = {
            'easy': 5,
            'medium': 8,
            'hard': 10
        }
        self.min_epochs_by_phase = {
            'easy': 30,
            'medium': 40,
            'hard': 50
        }

    def __call__(self, metrics: Dict[str, float], epoch: int, phase: str = 'easy') -> bool:
        """
        Check if training should stop based on a composite score of multiple metrics.

        Args:
            metrics: Dictionary containing validation metrics
            epoch: Current epoch number
            phase: Training phase ('easy', 'medium', 'hard')

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Calculate composite score based on phase
        if phase == 'easy':
            composite_score = (
                    0.4 * metrics['char_accuracy'] +
                    0.3 * metrics['end_char_accuracy'] +
                    0.2 * metrics['length_accuracy'] +
                    0.1 * (1.0 - metrics['char_ngram_loss'])
            )
        elif phase == 'medium':
            composite_score = (
                    0.3 * metrics['char_accuracy'] +
                    0.3 * metrics['end_char_accuracy'] +
                    0.3 * metrics['length_accuracy'] +
                    0.1 * (1.0 - metrics['char_ngram_loss'])
            )
        else:  # hard
            composite_score = (
                    0.25 * metrics['char_accuracy'] +
                    0.25 * metrics['end_char_accuracy'] +
                    0.3 * metrics['length_accuracy'] +
                    0.2 * (1.0 - metrics['char_ngram_loss'])
            )

        if self.best_composite_score is None:
            self.best_composite_score = composite_score
            return False

        improvement_threshold = {
            'easy': 0.01,
            'medium': 0.005,
            'hard': 0.003
        }[phase]

        absolute_improvement = composite_score - self.best_composite_score

        if absolute_improvement > improvement_threshold:
            self.best_composite_score = composite_score
            self.counter = 0
            return False

        if epoch < self.min_epochs_by_phase[phase]:
            return False

        self.counter += 1
        return self.counter >= self.patience_by_phase[phase]