from typing import Dict


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.best_composite_score = float('-inf')
        self.patience = patience
        self.counter = 0
        self.min_epoch = 30

    def __call__(self, metrics: Dict[str, float], epoch) -> bool:
        """
        Check if training should stop based on a composite score of multiple metrics.

        The composite score is weighted sum of:
        - Character accuracy (40%): How well individual characters are predicted
        - End character accuracy (30%): Accuracy of end characters which are crucial
        - Length accuracy (20%): How well the model predicts sequence lengths
        - Negative char n-gram loss (10%): How well local character patterns are preserved

        Args:
            metrics: Dictionary containing validation metrics
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        composite_score = (
                0.4 * float(metrics['val_char_accuracy']) +
                0.3 * float(metrics['val_end_char_accuracy']) +
                0.2 * float(metrics['val_length_accuracy']) +
                0.1 * (1.0 - float(metrics['val_char_ngram_loss']))
        )

        if composite_score > self.best_composite_score:
            self.best_composite_score = composite_score
            self.counter = 0
            return False
        elif epoch < self.min_epoch:
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience