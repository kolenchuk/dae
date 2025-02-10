import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
import logging
import os
import pandas as pd
import seaborn as sns
from dae import DAE
from model_save_utils import load_checkpoint_distributed
from text_dataset import TextDataset, pad_collate_fn


def validate_curriculum_model(
        model: torch.nn.Module,
        datasets: Dict[str, torch.utils.data.Dataset],
        batch_size: int = 64,
        device: str = "cuda",
        save_dir: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Validate the model separately on each difficulty level

    Args:
        model: The DAE model
        datasets: Dictionary mapping difficulty levels to datasets
        batch_size: Batch size for validation
        device: Device to run validation on
        save_dir: Directory to save validation results

    Returns:
        Dictionary containing validation metrics for each difficulty level
    """
    model.eval()
    results = {}

    for difficulty, dataset in datasets.items():
        logging.info(f"Validating on {difficulty} difficulty")

        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn
        )

        # Get metrics for this difficulty level
        metrics, error_dist, (predictions, targets) = validate_model(
            model, val_loader, dataset, device
        )

        # Analyze errors for this difficulty level
        error_analysis, error_details = analyze_error_distribution(
            predictions, targets,
            save_dir=os.path.join(save_dir, difficulty) if save_dir else None
        )

        results[difficulty] = {
            'metrics': metrics,
            'error_distribution': error_analysis,
            'error_details': error_details,
            'predictions': predictions,
            'targets': targets
        }

    if save_dir:
        save_curriculum_results(results, save_dir)
        plot_curriculum_comparisons(results, save_dir)

    return results


def save_curriculum_results(results: Dict, save_dir: str):
    """Save detailed curriculum validation results"""
    os.makedirs(save_dir, exist_ok=True)

    # Save overall metrics comparison
    metrics_comparison = {
        difficulty: result['metrics']
        for difficulty, result in results.items()
    }

    pd.DataFrame(metrics_comparison).to_csv(
        os.path.join(save_dir, 'metrics_comparison.csv')
    )

    # Save detailed results for each difficulty
    for difficulty, result in results.items():
        diff_dir = os.path.join(save_dir, difficulty)
        os.makedirs(diff_dir, exist_ok=True)

        # Save error distribution
        pd.DataFrame(
            result['error_distribution'].items(),
            columns=['Error_Type', 'Percentage']
        ).to_csv(os.path.join(diff_dir, 'error_distribution.csv'))

        # Save prediction examples
        save_prediction_examples(
            result['predictions'],
            result['targets'],
            os.path.join(diff_dir, 'prediction_examples.txt')
        )


def plot_curriculum_comparisons(results: Dict, save_dir: str):
    """Create comparative visualizations across difficulty levels"""
    # Metrics comparison plot
    plt.figure(figsize=(12, 6))
    metrics_data = {
        difficulty: result['metrics']
        for difficulty, result in results.items()
    }
    df_metrics = pd.DataFrame(metrics_data).transpose()

    sns.heatmap(
        df_metrics,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd'
    )
    plt.title('Metrics Comparison Across Difficulty Levels')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))
    plt.close()

    # Error distribution comparison
    plt.figure(figsize=(15, 6))
    error_data = {
        difficulty: result['error_distribution']
        for difficulty, result in results.items()
    }

    # Convert to DataFrame for plotting
    error_dfs = []
    for difficulty, dist in error_data.items():
        df = pd.DataFrame(dist.items(), columns=['Error_Type', 'Percentage'])
        df['Difficulty'] = difficulty
        error_dfs.append(df)

    df_errors = pd.concat(error_dfs)

    # Plot grouped bar chart
    sns.barplot(
        data=df_errors,
        x='Error_Type',
        y='Percentage',
        hue='Difficulty'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Error Distribution Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution_comparison.png'))
    plt.close()


def validate_model(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        dataset,
        device: str = "cuda"
) -> Tuple[Dict[str, float], Dict[str, int], Tuple[List[str], List[str]]]:
    """
    Validate the model using consistent metrics

    Args:
        model: The DAE model
        val_loader: Validation data loader
        dataset: TextDataset instance for decoding predictions
        device: Device to run validation on

    Returns:
        Tuple of (metrics, error_distributions, (predictions, targets))
    """
    model.eval()
    val_metrics = defaultdict(float)
    num_batches = 0
    all_predictions = []
    all_targets = []
    error_distributions = defaultdict(int)

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Get model predictions
            output = model(noisy)
            predictions = output.argmax(dim=-1)

            # Convert predictions and targets to text
            pred_texts = [dataset.indices_to_text(p.cpu().tolist()) for p in predictions]
            target_texts = [dataset.indices_to_text(t.cpu().tolist()) for t in clean]

            # Calculate metrics
            # Character accuracy
            char_correct = sum(
                sum(p == t for p, t in zip(pred, target))
                for pred, target in zip(pred_texts, target_texts)
            )
            total_chars = sum(len(t) for t in target_texts)
            val_metrics['char_accuracy'] += char_correct / total_chars if total_chars > 0 else 0

            # End character accuracy
            end_char_correct = sum(
                pred[-1] == target[-1] if pred and target else False
                for pred, target in zip(pred_texts, target_texts)
            )
            val_metrics['end_char_accuracy'] += end_char_correct / len(target_texts)

            # Length accuracy
            length_match = sum(
                len(pred) == len(target)
                for pred, target in zip(pred_texts, target_texts)
            )
            val_metrics['length_accuracy'] += length_match / len(target_texts)

            # Levenshtein distance
            lev_distances = [
                levenshtein_distance(pred, target)
                for pred, target in zip(pred_texts, target_texts)
            ]
            val_metrics['levenshtein'] += sum(lev_distances) / len(lev_distances)

            # Exact matches
            exact_matches = sum(p == t for p, t in zip(pred_texts, target_texts))
            val_metrics['exact_matches'] += exact_matches / len(target_texts)

            # Store predictions and targets
            all_predictions.extend(pred_texts)
            all_targets.extend(target_texts)

            # Update error distributions
            for pred, target in zip(pred_texts, target_texts):
                if pred != target:
                    error_type = categorize_error(pred, target)
                    error_distributions[error_type] += 1

            num_batches += 1

    # Average metrics
    for key in val_metrics:
        val_metrics[key] /= num_batches

    return val_metrics, error_distributions, (all_predictions, all_targets)


def analyze_error_distribution(
        predictions: List[str],
        targets: List[str],
        save_dir: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, str]]]]:
    """Analyze error distribution in model predictions"""
    error_analysis = defaultdict(int)
    error_details = defaultdict(list)

    for pred, target in zip(predictions, targets):
        if pred != target:
            # Categorize error
            error_type = categorize_error(pred, target)
            error_analysis[error_type] += 1
            error_details[error_type].append((target, pred))

    # Calculate percentages
    total_errors = sum(error_analysis.values())
    error_distribution = {
        k: (v / total_errors * 100) if total_errors > 0 else 0
        for k, v in error_analysis.items()
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_error_analysis(error_distribution, error_details, save_dir)

    return error_distribution, error_details


def save_prediction_examples(
        predictions: List[str],
        targets: List[str],
        output_file: str,
        max_examples: int = 50
):
    """Save prediction examples to file"""
    with open(output_file, 'w') as f:
        f.write("PREDICTION EXAMPLES:\n")
        f.write("-" * 60 + "\n")

        for target, pred in list(zip(targets, predictions))[:max_examples]:
            f.write(f"Target: {target}\n")
            f.write(f"Pred  : {pred}\n")
            if target != pred:
                f.write(f"Error Type: {categorize_error(pred, target)}\n")
                f.write(f"Levenshtein: {levenshtein_distance(pred, target)}\n")
            f.write("-" * 60 + "\n")


def save_error_analysis(
        error_distribution: Dict[str, float],
        error_details: Dict[str, List[Tuple[str, str]]],
        save_dir: str
):
    """Save error analysis results"""
    # Save error distribution plot
    plt.figure(figsize=(12, 6))

    sorted_errors = sorted(
        error_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )
    labels, values = zip(*sorted_errors)

    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Error Type Distribution')
    plt.ylabel('Percentage of Errors')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()

    # Save error examples
    with open(os.path.join(save_dir, 'error_examples.txt'), 'w') as f:
        for error_type, examples in error_details.items():
            f.write(f"\n{error_type.upper()} ERRORS:\n")
            f.write("-" * 60 + "\n")

            for target, pred in examples[:10]:  # Show first 10 examples of each type
                f.write(f"Target: {target}\n")
                f.write(f"Pred  : {pred}\n")
                f.write(f"Levenshtein: {levenshtein_distance(pred, target)}\n")
                f.write("-" * 60 + "\n")


def categorize_error(pred: str, target: str) -> str:
    """Categorize the type of error based on prediction and target"""
    if len(pred) != len(target):
        if len(pred) < len(target):
            return 'truncation'
        return 'expansion'

    diff_chars = sum(p != t for p, t in zip(pred, target))
    if diff_chars == 1:
        return 'single_char'

    if len(pred) >= 2 and len(target) >= 2:
        transpositions = sum(
            pred[i:i + 2] == target[i + 1] + target[i]
            for i in range(len(pred) - 1)
        )
        if transpositions > 0:
            return 'transposition'

    if pred[:-1] == target[:-1] and pred[-1] != target[-1]:
        return 'end_char'

    return 'complex'

def main():
    config = {
        'model_path': './models/dae_best_checkpoint',
        'batch_size':  128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    }

    model, metadata = load_checkpoint_distributed(
        config['model_path'],
        DAE,
        config['device']
    )
    model.eval()

    curriculum_datasets = {
        'easy': TextDataset('./data/curriculum_datasets/dae_dataset_easy.csv'),
        'medium': TextDataset('./data/curriculum_datasets/dae_dataset_medium.csv'),
        'hard': TextDataset('./data/curriculum_datasets/dae_dataset_hard.csv')
    }

    # Validate model across all difficulty levels
    results = validate_curriculum_model(
        model,
        curriculum_datasets,
        batch_size=64,
        device="cuda",
        save_dir="./reports/validation_results"
    )

if __name__ == "__main__":
    main()