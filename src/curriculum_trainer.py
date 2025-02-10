import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List
import logging

from early_stopping import EarlyStopping
from text_dataset import TextDataset, pad_collate_fn
from dae_trainer import DAETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CurriculumTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            base_config: Dict,
            curriculum_datasets: Dict[str, str],
            device: str = "cuda"
    ):
        self.model = model
        self.base_config = base_config
        self.device = device
        self.datasets = self._load_datasets(curriculum_datasets)
        self.current_trainer = None

    def _load_datasets(self, curriculum_datasets: Dict[str, str]) -> Dict[str, TextDataset]:
        """Load datasets for each difficulty level"""
        return {
            difficulty: TextDataset(path)
            for difficulty, path in curriculum_datasets.items()
        }

    def _create_trainer_for_phase(
            self,
            datasets: List[TextDataset],
            phase_config: Dict
    ) -> DAETrainer:
        """Create a trainer for the current phase"""
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)

        # Create data loaders
        train_loader = DataLoader(
            combined_dataset,
            batch_size=phase_config['batch_size'],
            shuffle=True,
            collate_fn=pad_collate_fn,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            combined_dataset,
            batch_size=phase_config['batch_size'],
            shuffle=False,
            collate_fn=pad_collate_fn,
            num_workers=4,
            pin_memory=True
        )

        # Create trainer
        return DAETrainer(
            model=self.model,
            dataset=datasets[0],  # Use first dataset for vocab
            config=phase_config
        ), train_loader, val_loader

    def train_curriculum(self):
        """Execute curriculum learning training process"""
        # Phase 1: Train on easy errors only
        logging.info("Starting Phase 1: Easy errors")
        phase1_config = {
            **self.base_config,
            'learning_rate': self.base_config['learning_rate'] * 1.5,
            'max_epochs': 50,
            'phase_name': 'easy'
        }
        trainer1, train_loader1, val_loader1 = self._create_trainer_for_phase(
            [self.datasets['easy']],
            phase1_config
        )
        metrics1 = self._train_phase(trainer1, train_loader1, val_loader1)

        # Phase 2: Train on easy + medium errors
        logging.info("Starting Phase 2: Easy + Medium errors")
        phase2_config = {
            **self.base_config,
            'learning_rate': self.base_config['learning_rate'],
            'max_epochs': 75,
            'phase_name': 'medium'
        }
        trainer2, train_loader2, val_loader2 = self._create_trainer_for_phase(
            [self.datasets['easy'], self.datasets['medium']],
            phase2_config
        )
        metrics2 = self._train_phase(trainer2, train_loader2, val_loader2)

        # Phase 3: Train on all errors
        logging.info("Starting Phase 3: All errors")
        phase3_config = {
            **self.base_config,
            'learning_rate': self.base_config['learning_rate'] * 0.5,
            'max_epochs': 100,
            'phase_name': 'hard'
        }
        trainer3, train_loader3, val_loader3 = self._create_trainer_for_phase(
            [self.datasets['easy'], self.datasets['medium'], self.datasets['hard']],
            phase3_config
        )
        metrics3 = self._train_phase(trainer3, train_loader3, val_loader3)

        return {
            'phase1_metrics': metrics1,
            'phase2_metrics': metrics2,
            'phase3_metrics': metrics3
        }

    def _train_phase(
            self,
            trainer: DAETrainer,
            train_loader: DataLoader,
            val_loader: DataLoader
    ) -> Dict:
        """Train a single phase of the curriculum"""
        phase_metrics = []

        early_stopping = EarlyStopping(patience=trainer.config['patience'])
        for epoch in range(trainer.config['max_epochs']):
            print(f'Epoch {epoch + 1}/{trainer.config["max_epochs"]}')
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)

            phase_metrics.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            trainer.save_checkpoint(
                trainer.config['save_dir'],
                epoch,
                train_metrics,
                val_metrics
            )

            print(f"train_metrics: {train_metrics}")
            print(f"val_metrics: {val_metrics}")
            print('-' * 50)
            if early_stopping(val_metrics, epoch, trainer.config['phase_name']):
                logger.info(f'Early stopping at epoch {epoch}. Best Score: {early_stopping.best_composite_score:.4f}')
                break

        return phase_metrics