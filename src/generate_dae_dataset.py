import json
import os
import random
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto
from real_words_errors import RealWorldErrors


class ErrorDifficulty(Enum):
    EASY = auto()      # Single character errors
    MEDIUM = auto()    # Two character errors
    HARD = auto()      # Complex errors

@dataclass
class ErrorPattern:
    difficulty: ErrorDifficulty
    weight: float
    generator: callable


class CurriculumDatasetGenerator:
    def __init__(self, real_world_errors_file: str = None):
        self.typing_patterns = {
            'layout_errors': {
                'a': ['s', 'q', 'w'],
                's': ['a', 'w', 'd'],
                'd': ['s', 'e', 'f'],
                'f': ['d', 'r', 'g'],
                'g': ['f', 't', 'h'],
                'h': ['g', 'y', 'j'],
                'j': ['h', 'u', 'k'],
                'k': ['j', 'i', 'l'],
                'l': ['k', 'o', 'p'],
                'z': ['x', 'a', 's'],
                'x': ['z', 's', 'c'],
                'c': ['x', 'd', 'v'],
                'v': ['c', 'f', 'b'],
                'b': ['v', 'g', 'n'],
                'n': ['b', 'h', 'm'],
                'm': ['n', 'j', 'k'],
                'q': ['w', 'a'],
                'w': ['q', 's', 'e'],
                'e': ['w', 'd', 'r'],
                'r': ['e', 'f', 't'],
                't': ['r', 'g', 'y'],
                'y': ['t', 'h', 'u'],
                'u': ['y', 'j', 'i'],
                'i': ['u', 'k', 'o'],
                'o': ['i', 'l', 'p'],
                'p': ['o', 'l']
            },

            # Типові закінчення слів, де часто виникають помилки
            'common_endings': {
                'ing': ['ing', 'ng', 'in', 'img'],
                'ion': ['ion', 'on', 'io', 'iom'],
                'ent': ['ent', 'ant', 'emt'],
                'able': ['able', 'abl', 'ible'],
                'tion': ['tion', 'ton', 'tiom']
            },

            # Помилки свайпу/автокорекції
            'swipe_patterns': {
                'pro': ['pro', 'pro', 'ore', 'pfo'],
                'max': ['max', 'mac', 'msx'],
                'mini': ['mini', 'mimi', 'moni'],
                'ultra': ['ultra', 'ulta', 'yltra'],
                'plus': ['plus', 'pkus', 'plys']
            },

            # Символи, які часто дублюються при швидкому наборі
            'double_chars': ['a', 'e', 'i', 'o', 'n', 'r', 's', 't', 'l', 'p'],

            # Групи символів, де часто пропускають букви
            'skip_patterns': ['ae', 'ao', 'ia', 'ie', 'io', 'oa', 'ue', 'ui']
        }

        self.error_patterns = self._initialize_error_patterns()
        self.real_world_errors = RealWorldErrors(real_world_errors_file)

    def _initialize_error_patterns(self) -> Dict[ErrorDifficulty, List[ErrorPattern]]:
        """Initialize error patterns with difficulty levels"""
        return {
            ErrorDifficulty.EASY: [
                ErrorPattern(
                    ErrorDifficulty.EASY,
                    0.4,
                    self.generate_single_typo
                ),
                ErrorPattern(
                    ErrorDifficulty.EASY,
                    0.3,
                    self.generate_single_deletion
                ),
                ErrorPattern(
                    ErrorDifficulty.EASY,
                    0.3,
                    self.generate_single_insertion
                )
            ],
            ErrorDifficulty.MEDIUM: [
                ErrorPattern(
                    ErrorDifficulty.MEDIUM,
                    0.4,
                    self.generate_transposition_error
                ),
                ErrorPattern(
                    ErrorDifficulty.MEDIUM,
                    0.3,
                    self.generate_double_char_error
                ),
                ErrorPattern(
                    ErrorDifficulty.MEDIUM,
                    0.3,
                    self.generate_end_truncation
                ),
                ErrorPattern(
                    ErrorDifficulty.MEDIUM,
                    0.3,
                    self.generate_ending_error
                ),
                ErrorPattern(
                    ErrorDifficulty.MEDIUM,
                    0.4,
                    self.generate_real_world_error
                )
            ],
            ErrorDifficulty.HARD: [
                ErrorPattern(
                    ErrorDifficulty.HARD,
                    0.4,
                    self.generate_swipe_error
                ),
                ErrorPattern(
                    ErrorDifficulty.HARD,
                    0.3,
                    self.generate_multiple_errors
                ),
                ErrorPattern(
                    ErrorDifficulty.HARD,
                    0.3,
                    self.generate_autocorrect_error
                )
            ]
        }

    def generate_single_typo(self, word: str) -> str:
        """Generate a single character typo"""
        if len(word) < 2:
            return word
        pos = random.randint(0, len(word) - 1)
        chars = list(word)
        chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(chars)

    def generate_single_deletion(self, word: str) -> str:
        """Delete a single character"""
        if len(word) < 2:
            return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    def generate_single_insertion(self, word: str) -> str:
        """Insert a single character"""
        pos = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos:]

    def generate_keyboard_typo(self, word: str) -> str:
        """Генерує помилки на основі сусідніх клавіш."""
        if len(word) < 4:
            return word

        pos = random.randint(1, len(word) - 2)  # Не чіпаємо перший і останній символи
        char = word[pos]

        if char in self.typing_patterns['layout_errors']:
            new_char = random.choice(self.typing_patterns['layout_errors'][char])
            return word[:pos] + new_char + word[pos + 1:]
        return word

    def generate_ending_error(self, word: str) -> str:
        """Генерує помилки в типових закінченнях слів."""
        for ending, variants in self.typing_patterns['common_endings'].items():
            if word.endswith(ending):
                new_ending = random.choice(variants)
                return word[:-len(ending)] + new_ending
        return word

    def generate_swipe_error(self, word: str) -> str:
        """Генерує помилки свайпу."""
        for pattern, variants in self.typing_patterns['swipe_patterns'].items():
            if pattern in word:
                new_pattern = random.choice(variants)
                return word.replace(pattern, new_pattern)
        return word

    def generate_double_char_error(self, word: str) -> str:
        """Генерує помилки з подвоєнням символів."""
        candidates = []
        for i, char in enumerate(word):
            if char in self.typing_patterns['double_chars']:
                candidates.append(i)

        if candidates:
            pos = random.choice(candidates)
            return word[:pos] + word[pos] + word[pos:]
        return word

    ### ---
    def generate_deletion_error(self, word: str) -> str:
        """Видаляє випадковий символ."""
        if len(word) <= 3:
            return word
        pos = random.randint(1, len(word) - 2)
        return word[:pos] + word[pos + 1:]

    def generate_transposition_error(self, word: str) -> str:
        """Переставляє місцями два сусідніх символи."""
        if len(word) < 4:
            return word
        pos = random.randint(1, len(word) - 2)
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

    def generate_autocorrect_error(self, word: str) -> str:
        """Імітація помилок автокорекції."""
        common_mistakes = {
            "iphone": "ihpone",
            "samsung": "samsang",
            "google": "googel"
        }
        return common_mistakes.get(word, word)

    def generate_end_truncation(self, word: str) -> str:
        """
        Generate errors by truncating the end of words, which is common in mobile searches.
        Examples: samsung -> samsun, nokia -> noki, iphone -> iphon
        """
        if len(word) <= 4:  # Don't truncate very short words
            return word

        # Common endings that often get truncated
        common_truncations = {
            'samsung': ['samsun', 'samsu'],
            'nokia': ['noki'],
            'xiaomi': ['xiaom'],
            'huawei': ['huawe'],
            'iphone': ['iphon'],
            'google': ['googl'],
            'realme': ['realm'],
            'redmi': ['redm'],
            'oppo': ['opp'],
            'vivo': ['viv']
        }

        # If it's a known word, use predefined truncation
        if word.lower() in common_truncations:
            return random.choice(common_truncations[word.lower()])

        # For other words, intelligently truncate based on patterns
        if word.endswith('ing'):
            return word[:-1]  # Remove last 'g'
        elif word.endswith(('ung', 'ond', 'one', 'mi')):
            return word[:-1]  # Remove last character
        elif len(word) >= 6:
            # Randomly truncate 1 or 2 characters for longer words
            truncate_length = random.choice([1, 2])
            return word[:-truncate_length]

        return word

    def generate_real_world_error(self, word: str) -> str:
        """Generate errors based on real-world patterns"""
        common_errors = self.real_world_errors.get_common_errors(word)
        if common_errors:
            return random.choice(common_errors)
        return word

    def generate_multiple_errors(self, word: str) -> str:
        """
        Generate multiple errors in a word by combining different error types.
        Focuses on realistic mobile typing scenarios.
        """
        if len(word) < 4:
            return word

        # Select number of errors (2 or 3) based on word length
        num_errors = 3 if len(word) >= 6 else 2

        # List of available error generators, excluding this one to prevent recursion
        basic_error_generators = [
            self.generate_single_typo,
            self.generate_keyboard_typo,
            self.generate_single_deletion,
            self.generate_single_insertion,
            self.generate_ending_error,
            self.generate_swipe_error,
            self.generate_double_char_error,
            self.generate_transposition_error,
            self.generate_end_truncation
        ]

        # Make a copy of the word to modify
        result = word
        used_positions = set()

        # Apply multiple error transformations
        for _ in range(num_errors):
            # Choose a random error generator
            error_gen = random.choice(basic_error_generators)

            # Apply the error transformation
            new_result = error_gen(result)

            # If the error generator didn't change the word, try another one
            attempts = 0
            while new_result == result and attempts < 3:
                error_gen = random.choice(basic_error_generators)
                new_result = error_gen(result)
                attempts += 1

            result = new_result

            # Check if the word is still somewhat recognizable
            if levenshtein_distance(result, word) > len(word) // 2:
                break

        return result

    def generate_dataset_for_difficulty(
            self,
            words: List[str],
            difficulty: ErrorDifficulty,
            variations_per_word: int
    ) -> List[Tuple[str, str, str]]:
        """Generate dataset entries for specific difficulty level"""
        dataset = []
        patterns = self.error_patterns[difficulty]

        for word in words:
            word_variations = set()
            attempts = 0
            max_attempts = variations_per_word * 4

            while len(word_variations) < variations_per_word and attempts < max_attempts:
                # Select error pattern based on weights
                weights = [p.weight for p in patterns]
                pattern = random.choices(patterns, weights=weights)[0]

                noisy_word = pattern.generator(word)

                if (levenshtein_distance(noisy_word, word) > 0 and
                        noisy_word not in word_variations):
                    word_variations.add(noisy_word)
                attempts += 1

            for noisy_word in word_variations:
                dataset.append((noisy_word, word, difficulty.name.lower()))

        return dataset

    def generate_curriculum_dataset(
            self,
            input_file: str,
            output_dir: str,
            variations_per_word: Dict[ErrorDifficulty, int] = None
    ):
        """Generate curriculum learning datasets"""
        if variations_per_word is None:
            variations_per_word = {
                ErrorDifficulty.EASY: 10,
                ErrorDifficulty.MEDIUM: 7,
                ErrorDifficulty.HARD: 5
            }

        # Read words
        with open(input_file, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f.readlines() if line.strip()]

        # Generate datasets for each difficulty
        datasets = {}
        for difficulty in ErrorDifficulty:
            datasets[difficulty] = self.generate_dataset_for_difficulty(
                words,
                difficulty,
                variations_per_word[difficulty]
            )

        # Save separate datasets
        os.makedirs(output_dir, exist_ok=True)

        for difficulty, dataset in datasets.items():
            df = pd.DataFrame(dataset, columns=["noisy_input", "clean_output", "difficulty"])

            # Split into train/test
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)

            # Add split column
            train_df['split'] = 'train'
            test_df['split'] = 'test'

            # Combine and save
            final_df = pd.concat([train_df, test_df])
            output_file = os.path.join(output_dir, f"dae_dataset_{difficulty.name.lower()}.csv")
            final_df.to_csv(output_file, index=False, encoding="utf-8")

            print(f"{difficulty.name} dataset generated with {len(dataset)} records")
            print(f"Saved to: {output_file}")
            print("Sample errors:")
            for noisy, clean, _ in dataset[:5]:
                print(f"{clean} -> {noisy}")
            print()


def update_patterns_from_logs(log_file: str, pattern_file: str):
    """Process logs to extract and update error patterns"""
    real_world_errors = RealWorldErrors(pattern_file)
    updates_count = 0
    errors_count = 0

    print(f"Processing log file: {log_file}")

    # Process your logs
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON and validate structure
                log_entry = json.loads(line.strip())

                if not isinstance(log_entry, dict):
                    print(f"Line {line_num}: Invalid log entry format - expected dictionary")
                    errors_count += 1
                    continue

                # Verify required fields exist
                if 'correct' not in log_entry or 'error' not in log_entry:
                    print(f"Line {line_num}: Missing required fields 'correct' or 'error'")
                    errors_count += 1
                    continue

                # Get and validate the values
                correct = str(log_entry['correct']).lower().strip()
                error = str(log_entry['error']).lower().strip()

                if not correct or not error:
                    print(f"Line {line_num}: Empty correct or error value")
                    errors_count += 1
                    continue

                # Update patterns
                if correct not in real_world_errors.error_patterns:
                    real_world_errors.error_patterns[correct] = []
                if error not in real_world_errors.error_patterns[correct]:
                    real_world_errors.error_patterns[correct].append(error)
                    updates_count += 1

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON parsing error - {str(e)}")
                errors_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error - {str(e)}")
                errors_count += 1

    # Save updated patterns
    real_world_errors.save_patterns(pattern_file)

    print(f"\nProcessing complete:")
    print(f"- Added {updates_count} new error patterns")
    print(f"- Encountered {errors_count} errors during processing")

    return updates_count, errors_count


def main():
    config = {
        "input_file": "./data/unique_words.txt",
        "output_dir": "./data/curriculum_datasets",
        "variations_per_word": {
            ErrorDifficulty.EASY: 10,
            ErrorDifficulty.MEDIUM: 7,
            ErrorDifficulty.HARD: 5
        }
    }

    generator = CurriculumDatasetGenerator()
    generator.generate_curriculum_dataset(**config)
    update_patterns_from_logs(
        './logs/search_logs.json',
        './data/real_world_patterns.json'
    )

if __name__ == "__main__":
    main()