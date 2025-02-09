import torch
from torch.utils.data import DataLoader
from src.dae_trainer import DAETrainer
from src.early_stopping import EarlyStopping
from text_dataset import TextDataset, pad_collate_fn
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

        'save_dir': '../models',
        'dataset_path': '../data/dae_dataset.csv',
        'log_interval': 10,

        'val_interval': 100,
        'plot_interval': 500,

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

    dataset = TextDataset(config['dataset_path'])
    config['pad_idx'] = dataset.char_to_idx['<PAD>']

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = DAE(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx'],
        max_len=config['max_len']
    ).to(device)

    trainer = DAETrainer(
        model=model,
        dataset=dataset,
        config=config
    )

    early_stopping = EarlyStopping(patience=config['patience'])

    for epoch in range(config['max_epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        logger.info(f'Epoch {epoch + 1}/{config["max_epochs"]} - '
                    f'Train Loss: {val_metrics["train_loss"]:.4f}, '
                    f'Val Loss: {val_metrics["val_loss"]:.4f}')

        trainer.save_checkpoint(
            config['save_dir'],
            epoch,
            {**train_metrics, **val_metrics}
        )
        print(val_metrics)
        if early_stopping(val_metrics, epoch):
            logger.info(f'Early stopping at epoch {epoch}. Best Score: {early_stopping.best_composite_score:.4f}')
            break

    trainer.plot_metrics(config['save_dir'])


if __name__ == "__main__":
    main()