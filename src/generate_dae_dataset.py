import os
import random
import pandas as pd
from Levenshtein import distance as levenshtein_distance

TYPING_PATTERNS = {
    # Типові помилки розкладки
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


def generate_keyboard_typo(word: str) -> str:
    """Генерує помилки на основі сусідніх клавіш."""
    if len(word) < 4:
        return word

    pos = random.randint(1, len(word) - 2)  # Не чіпаємо перший і останній символи
    char = word[pos]

    if char in TYPING_PATTERNS['layout_errors']:
        new_char = random.choice(TYPING_PATTERNS['layout_errors'][char])
        return word[:pos] + new_char + word[pos + 1:]
    return word


def generate_ending_error(word: str) -> str:
    """Генерує помилки в типових закінченнях слів."""
    for ending, variants in TYPING_PATTERNS['common_endings'].items():
        if word.endswith(ending):
            new_ending = random.choice(variants)
            return word[:-len(ending)] + new_ending
    return word


def generate_swipe_error(word: str) -> str:
    """Генерує помилки свайпу."""
    for pattern, variants in TYPING_PATTERNS['swipe_patterns'].items():
        if pattern in word:
            new_pattern = random.choice(variants)
            return word.replace(pattern, new_pattern)
    return word


def generate_double_char_error(word: str) -> str:
    """Генерує помилки з подвоєнням символів."""
    candidates = []
    for i, char in enumerate(word):
        if char in TYPING_PATTERNS['double_chars']:
            candidates.append(i)

    if candidates:
        pos = random.choice(candidates)
        return word[:pos] + word[pos] + word[pos:]
    return word

def generate_deletion_error(word: str) -> str:
    """Видаляє випадковий символ."""
    if len(word) <= 3:
        return word
    pos = random.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1:]

def generate_transposition_error(word: str) -> str:
    """Переставляє місцями два сусідніх символи."""
    if len(word) < 4:
        return word
    pos = random.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

def generate_autocorrect_error(word: str) -> str:
    """Імітація помилок автокорекції."""
    common_mistakes = {
        "iphone": "ihpone",
        "samsung": "samsang",
        "google": "googel"
    }
    return common_mistakes.get(word, word)

def introduce_error(word: str) -> str:
    """Головна функція генерації помилок."""
    error_functions = [
        (generate_keyboard_typo, 0.25),
        (generate_ending_error, 0.2),
        (generate_swipe_error, 0.15),
        (generate_double_char_error, 0.1),
        (generate_deletion_error, 0.15),
        (generate_transposition_error, 0.1),
        (generate_autocorrect_error, 0.05)
    ]

    num_errors = 1 if len(word) <= 5 else random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

    result = word
    for _ in range(num_errors):
        func, _ = random.choices(error_functions, weights=[w for _, w in error_functions])[0]
        result = func(result)

    if len(result) < 4:
        return word
    return result

def generate_dataset(input_file: str, output_file: str):
    """Генерація датасету з помилками."""
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines() if line.strip()]

    dataset = set()

    for word in words:
        variations_count = 20 if len(word) >= 6 else 15
        word_variations = set()
        attempts = 0
        max_attempts = variations_count * 4

        while len(word_variations) < variations_count and attempts < max_attempts:
            noisy_word = introduce_error(word)
            if (levenshtein_distance(noisy_word, word) > 0 and
                noisy_word not in word_variations and
                len(noisy_word) >= max(3, len(word) * 0.75)):
                word_variations.add(noisy_word)
            attempts += 1

        for noisy_word in word_variations:
            dataset.add((noisy_word, word))

    dataset = list(dataset)
    random.shuffle(dataset)

    split_ratio = 0.8
    split_index = int(len(dataset) * split_ratio)

    dataset = [(noisy, clean, "train" if i < split_index else "test") for i, (noisy, clean) in enumerate(dataset)]
    df = pd.DataFrame(dataset, columns=["noisy_input", "clean_output", "split"])
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Dataset generated with {len(dataset)} records")
    print(f"Train samples: {split_index}")
    print(f"Test samples: {len(dataset) - split_index}")
    for noisy, clean, split in dataset[:10]:
        print(f"{clean} -> {noisy} ({split})")

if __name__ == "__main__":
    # Configuration
    config = {
        "input_file": "../data/unique_words.txt",
        "output_file": "../data/dae_dataset.csv",
    }

    generate_dataset(**config)