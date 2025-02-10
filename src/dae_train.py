import torch

from analyze_curriculum_metrics import analyze_curriculum_metrics
from curriculum_trainer import CurriculumTrainer
from text_dataset import TextDataset
from dae import DAE
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config = {
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_len': 20,

        'learning_rate': 2e-4,
        'label_smoothing': 0.05,
        'warmup_steps': 200,
        'batch_size': 64,
        'max_epochs': 300,
        'patience': 5,
        'max_grad_norm': 1.0,
        'use_amp': True,

        'save_dir': './models',
        'val_interval': 100,

         'optimizer': {
            'betas': (0.9, 0.98),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'warmup_steps': 100
        },

        'lr_schedule': {
            'warmup_steps': 200,
            'decay_factor': 0.98,
            'decay_steps': 500
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')

    os.makedirs(config['save_dir'], exist_ok=True)

    config['pad_idx'] = 0

    model = DAE(
        vocab_size=29,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx'],
        max_len=config['max_len']
    ).to(device)

    curriculum_datasets = {
        'easy': './data/curriculum_datasets/dae_dataset_easy.csv',
        'medium': './data/curriculum_datasets/dae_dataset_medium.csv',
        'hard': './data/curriculum_datasets/dae_dataset_hard.csv'
    }

    curriculum_trainer = CurriculumTrainer(
        model=model,
        base_config=config,
        curriculum_datasets=curriculum_datasets
    )

    metrics = curriculum_trainer.train_curriculum()

    save_dir = './reports'
    summary = analyze_curriculum_metrics(metrics, save_dir)
    print("Summary Report:", summary)


if __name__ == "__main__":
    main()