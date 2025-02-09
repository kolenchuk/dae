import torch
import logging
from model_validator import ModelValidator
from src.model_save_utils import load_checkpoint_distributed
from text_dataset import TextDataset, pad_collate_fn
from dae import DAE
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model(model, val_loader, dataset, device="cuda"):
    """
    Функція для перевірки якості навчання моделі на валідаційному наборі.

    - `model`: нейромережа (DAE)
    - `val_loader`: DataLoader для валідаційного набору
    - `dataset`: об'єкт dataset для конвертації індексів у текст
    - `device`: 'cuda' або 'cpu'
    """

    model.eval()
    total_levenshtein = 0
    total_bleu = 0
    total_correct = 0
    total_samples = 0
    smoother = SmoothingFunction().method1

    results = []

    with torch.no_grad():
        for noisy, clean in tqdm(val_loader, desc="Validating"):
            noisy, clean = noisy.to(device), clean.to(device)

            output = model(noisy)
            output_probs = F.softmax(output / 1.0, dim=-1)
            predictions = output_probs.argmax(dim=-1)

            noisy_texts = [dataset.indices_to_text(seq) for seq in noisy.cpu().tolist()]
            predicted_texts = [dataset.indices_to_text(seq) for seq in predictions.cpu().tolist()]
            target_texts = [dataset.indices_to_text(seq) for seq in clean.cpu().tolist()]

            for i in range(len(noisy_texts)):
                lev_dist = levenshtein_distance(predicted_texts[i], target_texts[i])
                bleu_score = sentence_bleu(
                    [list(target_texts[i])],
                    list(predicted_texts[i]),
                    smoothing_function=smoother
                ) * 100
                is_correct = 1 if predicted_texts[i] == target_texts[i] else 0

                total_levenshtein += lev_dist
                total_bleu += bleu_score
                total_correct += is_correct
                total_samples += 1

                results.append({
                    "noisy": noisy_texts[i],
                    "predicted": predicted_texts[i],
                    "target": target_texts[i],
                    "levenshtein_distance": lev_dist,
                    "bleu_score": bleu_score
                })

    avg_lev_dist = total_levenshtein / total_samples
    avg_bleu = total_bleu / total_samples
    accuracy = (total_correct / total_samples) * 100

    print("\nValidation Results:")
    print(f"Avg Levenshtein Distance: {avg_lev_dist:.2f}")
    print(f"Avg BLEU Score: {avg_bleu:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")

    with open("../reports/validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n🔍 Результати збережено у `./reports/validation_results.json`")

    return avg_lev_dist, avg_bleu, accuracy

def analyze_error_distribution(results):
    error_types = defaultdict(int)
    error_examples = defaultdict(list)

    for result in results:
        if not result['is_correct']:
            noisy = result['noisy']
            corrected = result['corrected']
            expected = result['expected']

            # Categorize error
            if len(corrected) != len(expected):
                error_type = 'length_mismatch'
            elif all(c == 'p' for c in corrected):
                error_type = 'p_repetition'
            elif corrected[0] != expected[0]:
                error_type = 'first_char_error'
            elif levenshtein_distance(corrected, expected) == 1:
                error_type = 'single_char_error'
            else:
                error_type = 'multiple_errors'

            error_types[error_type] += 1
            error_examples[error_type].append({
                'noisy': noisy,
                'corrected': corrected,
                'expected': expected
            })

    return dict(error_types), dict(error_examples)

def main():
    config = {
        'model_path': '../models/dae_best_checkpoint',
        'dataset_path': '../data/dae_dataset.csv',
        'batch_size':  128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    }
    dataset = TextDataset(config['dataset_path'])

    model, metadata = load_checkpoint_distributed(
        config['model_path'],
        DAE,
        config['device']
    )
    model.eval()

    val_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    validate_model(model, val_loader, dataset, config['device'])

    validator = ModelValidator(model, metadata['char_to_idx'], metadata['idx_to_char'], config['device'])

    validator.run_validation_suite()

    results = validator.validate_on_test_cases([
        ("samung", "samsung"),
        ("aplple", "apple"),
        ("galxy", "galaxy"),
        ("nkia", "nokia"),
        ("xiomi", "xiaomi"),
        ("opppo", "oppo"),
        ("vivo", "vivo"),
        ("hwuaei", "huawei"),
        ("readmi", "redmi"),
        ("ulta", "ultra"),
        ("proo", "pro"),
        ("max", "max"),
        ("pluss", "plus"),
        ("lite", "lite"),
        ("nokua", "nokia"),
        ("sansung", "samsung"),
        ("realne", "realme"),
        ("galaxxy", "galaxy"),
        ("iphonne", "iphone"),
        ("xiaaomi", "xiaomi")
    ])

    error_types, error_examples = analyze_error_distribution(results['detailed_results'])

    for error_type, count in error_types.items():
        logger.info(f"{error_type}: {count} occurrences")
        logger.info("Examples:")
        for example in error_examples[error_type][:3]:
            logger.info(f"  {example['noisy']} -> {example['corrected']} (expected: {example['expected']})")

    results_file = '../reports/additional_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'error_analysis': {
                'types': error_types,
                'examples': error_examples
            }
        }, f, indent=2)

if __name__ == "__main__":
    main()