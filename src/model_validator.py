import torch
from typing import List, Dict, Tuple
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from text_utils import text_to_indices, indices_to_text
from pathlib import Path


class ModelValidator:
    def __init__(self, model, char_to_idx, idx_to_char, device="cuda"):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
        self.model.eval()

        self.valid_words = set(Path('../data/unique_words.txt').read_text().splitlines())

        self.metrics = defaultdict(list)

    def correct_word(self, noisy_word: str) -> Tuple[str, Dict]:
        """
        Correct a single word and return correction with detailed metrics.
        """
        with torch.no_grad():
            indices = text_to_indices(noisy_word.lower(), self.char_to_idx)
            input_tensor = torch.tensor([indices], device=self.device)

            output = self.model(input_tensor)
            predictions = output.argmax(dim=-1)

            corrected = indices_to_text(predictions[0].tolist(), self.idx_to_char)

            metrics = {
                'input_length': len(noisy_word),
                'output_length': len(corrected),
                'levenshtein_distance': levenshtein_distance(noisy_word, corrected),
                'is_valid_word': corrected in self.valid_words,
                'changed': noisy_word != corrected,
                'confidence': float(torch.softmax(output, dim=-1).max())
            }

            return corrected, metrics

    def validate_on_test_cases(self, test_cases: List[Tuple[str, str]]) -> Dict:
        """
        Run validation on a list of test cases and compute comprehensive metrics.
        """
        results = []
        metrics = defaultdict(list)
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for noisy, expected in test_cases:
            corrected, case_metrics = self.correct_word(noisy)

            is_correct = corrected == expected
            metrics['exact_match'].append(is_correct)
            metrics['levenshtein'].append(case_metrics['levenshtein_distance'])
            metrics['valid_word'].append(case_metrics['is_valid_word'])
            metrics['confidence'].append(case_metrics['confidence'])

            if not is_correct:
                error_key = f"{noisy}->{corrected}"
                confusion_matrix[expected][error_key] += 1

            results.append({
                'noisy': noisy,
                'corrected': corrected,
                'expected': expected,
                'is_correct': is_correct,
                **case_metrics
            })

        summary = {
            'accuracy': sum(metrics['exact_match']) / len(metrics['exact_match']),
            'mean_levenshtein': sum(metrics['levenshtein']) / len(metrics['levenshtein']),
            'valid_word_rate': sum(metrics['valid_word']) / len(metrics['valid_word']),
            'mean_confidence': sum(metrics['confidence']) / len(metrics['confidence']),
            'total_cases': len(test_cases)
        }

        error_patterns = self.analyze_error_patterns(results)

        return {
            'summary': summary,
            'detailed_results': results,
            'error_patterns': error_patterns,
            'confusion_matrix': dict(confusion_matrix)
        }

    def analyze_error_patterns(self, results: List[Dict]) -> Dict:
        """
        Analyze patterns in correction errors.
        """
        patterns = {
            'character_substitutions': defaultdict(int),
            'length_changes': defaultdict(int),
            'position_errors': defaultdict(int),
            'common_mistakes': defaultdict(int)
        }

        for result in results:
            if not result['is_correct']:
                noisy = result['noisy']
                corrected = result['corrected']
                expected = result['expected']

                # Analyze character substitutions
                for i, (n, e) in enumerate(zip(noisy, expected)):
                    if i < len(corrected) and corrected[i] != e:
                        patterns['character_substitutions'][f"{e}->{corrected[i]}"] += 1

                # Analyze length changes
                length_diff = len(corrected) - len(expected)
                patterns['length_changes'][length_diff] += 1

                # Analyze position-specific errors
                for i, (c, e) in enumerate(zip(corrected, expected)):
                    if c != e:
                        patterns['position_errors'][i] += 1

                # Track common incorrect corrections
                patterns['common_mistakes'][f"{noisy}->{corrected}"] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        return {k: dict(v) for k, v in patterns.items()}

    def run_validation_suite(self) -> None:
        """
        Run comprehensive validation suite and output detailed report.
        """
        standard_cases = [
            ("samung", "samsung"),
            ("iphoone", "iphone"),
            ("galxy", "galaxy"),
            ("nkia", "nokia"),
            ("xiomi", "xiaomi")
        ]

        keyboard_cases = [
            ("sansung", "samsung"),
            ("nokua", "nokia"),
            ("iphpne", "iphone"),
            ("galaxu", "galaxy")
        ]

        brand_cases = [
            ("hwuaei", "huawei"),
            ("realmy", "realme"),
            ("redni", "redmi"),
            ("opppo", "oppo")
        ]

        complex_cases = [
            ("samsungg", "samsung"),
            ("iphonne", "iphone"),
            ("galexy", "galaxy"),
            ("nokkia", "nokia")
        ]

        results = {
            'standard': self.validate_on_test_cases(standard_cases),
            'keyboard': self.validate_on_test_cases(keyboard_cases),
            'brand': self.validate_on_test_cases(brand_cases),
            'complex': self.validate_on_test_cases(complex_cases)
        }

        self._generate_validation_report(results)

    def _generate_validation_report(self, results: Dict) -> None:
        """
        Generate and save detailed validation report.
        """
        report = []

        total_cases = sum(res['summary']['total_cases'] for res in results.values())
        total_correct = sum(
            res['summary']['accuracy'] * res['summary']['total_cases']
            for res in results.values()
        )

        report.append("=== DAE Model Validation Report ===\n")
        report.append(f"Total Test Cases: {total_cases}")
        report.append(f"Overall Accuracy: {total_correct / total_cases:.2%}\n")

        for category, res in results.items():
            report.append(f"\n{category.title()} Тест кейси ")
            report.append(f"Cases: {res['summary']['total_cases']}")
            report.append(f"Accuracy: {res['summary']['accuracy']:.2%}")
            report.append(f"Mean Levenshtein Distance: {res['summary']['mean_levenshtein']:.2f}")
            report.append(f"Valid Word Rate: {res['summary']['valid_word_rate']:.2%}")
            report.append("\nTop Errors:")

            mistakes = sorted(
                res['error_patterns']['common_mistakes'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for mistake, count in mistakes:
                report.append(f"  {mistake}: {count} times")

        report_text = '\n'.join(report)
        Path('../reports/validation_report.txt').write_text(report_text)
        print(report_text)