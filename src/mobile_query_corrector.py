from transformers import pipeline
import torch
import Levenshtein
from collections import defaultdict
from datasets import Dataset


class MobileQueryCorrector:
    def __init__(self, text_corrector):
        self.text_corrector = text_corrector

        # Common mobile device terms and their misspellings
        self.brand_dictionary = {
            "Samsung": ["samsng", "samung", "smsng"],
            "iPhone": ["iphne", "ifone", "iphn"],
            "Galaxy": ["galxy", "galaxi", "glaxy"],
            "Xiaomi": ["xiomi", "xaomi", "xiami"],
            "Huawei": ["huwei", "huawai", "huawei"],
            "OnePlus": ["1plus", "onplus", "1+", "1 +", "one plus", "1+ "],
            "Google": ["googl", "gogle", "googs"],
            "Pixel": ["pixl", "piksel", "pixl"],
            "Pro": ["pr", "po"],
            "Plus": ["pls", "pluss"],
            "Ultra": ["ultr", "ulta"],
            "Max": ["mx", "maks"],
            "Note": ["nt", "not"],
            "Nord": ["nrd", "nort"],
            "Redmi": ["rdmi", "redmi", "redm"]
        }

        # Create reverse mapping for quick lookup
        self.reverse_brand_dict = {}
        for correct, variants in self.brand_dictionary.items():
            for variant in variants + [correct.lower()]:
                self.reverse_brand_dict[variant.strip()] = correct

    def _is_model_number(self, text):
        """Check if text contains a model number pattern"""
        return any(char.isdigit() for char in text)

    def _preprocess_text(self, text):
        """Split text into tokens while preserving model numbers"""
        # Special handling for OnePlus variants
        text = text.replace("1+", "OnePlus").replace("1 +", "OnePlus")

        tokens = []
        current_token = ""

        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens

    def _correct_brand_terms(self, word):
        """Correct common brand misspellings using the dictionary"""
        # Don't correct model numbers
        if self._is_model_number(word) and not any(variant in word.lower() for variant in ["1+", "1 +"]):
            return word

        word_lower = word.lower()

        # Direct match in reverse dictionary
        if word_lower in self.reverse_brand_dict:
            return self.reverse_brand_dict[word_lower]

        # Find closest match using Levenshtein distance
        min_distance = float('inf')
        closest_word = word

        # Check against all correct brand terms
        for correct_word in self.brand_dictionary.keys():
            distance = Levenshtein.distance(word_lower, correct_word.lower())
            if distance < min_distance and distance <= 2:
                min_distance = distance
                closest_word = correct_word

        return closest_word if min_distance <= 2 else word

    def refine_spelling_batch(self, texts):
        """Process multiple queries efficiently"""
        # For now, process sequentially to maintain correction quality
        # Future optimization could involve more sophisticated batching
        return [self.refine_spelling(text) for text in texts]

    def refine_spelling(self, text):
        """Main method to correct spelling in mobile phone queries"""
        # Split into tokens
        tokens = self._preprocess_text(text)

        # First pass: identify brands and model numbers
        token_types = []  # 'brand', 'model', or None
        for token in tokens:
            if self._is_model_number(token):
                token_types.append('model')
            else:
                # Check if it's a known brand or variant
                is_brand = False
                for brand in self.brand_dictionary.keys():
                    if (token.lower() in [b.lower() for b in self.brand_dictionary[brand]] or
                            token.lower() == brand.lower()):
                        token_types.append('brand')
                        is_brand = True
                        break
                if not is_brand:
                    token_types.append(None)

        # Process each token with context awareness
        corrected_tokens = []
        brand_seen = False

        for i, (token, token_type) in enumerate(zip(tokens, token_types)):
            if token_type == 'model':
                corrected_tokens.append(token)
            elif token_type == 'brand':
                if not brand_seen or token.lower() != corrected_tokens[-1].lower():
                    corrected_token = self._correct_brand_terms(token)
                    corrected_tokens.append(corrected_token)
                    brand_seen = True
            else:
                # Try brand-specific correction first
                corrected_token = self._correct_brand_terms(token)

                # If no brand-specific correction and token is long enough
                if corrected_token == token and len(token) > 2:
                    bart_output = self.text_corrector(
                        token,
                        max_length=len(token) + 3,
                        min_length=max(len(token) - 1, 1),
                        num_beams=4,
                        do_sample=False,
                        length_penalty=0.2
                    )[0]["generated_text"].strip()

                    # Only use BART correction if it's similar enough
                    if Levenshtein.distance(token.lower(), bart_output.lower()) <= 2:
                        corrected_token = self._correct_brand_terms(bart_output)

                    # Don't repeat brand names
                    if corrected_token.lower() in [t.lower() for t in corrected_tokens]:
                        corrected_token = token

                corrected_tokens.append(corrected_token)

        # Combine tokens
        result = " ".join(corrected_tokens)

        # Validate output
        if len(result.split()) < len(text.split()) / 2:
            return text

        return result


# Initialize corrector
# corrector = MobileQueryCorrector()
#
# # Test cases
# test_queries = [
#     "Samsung Galaxy A14",
#     "Samsng Galaxy A14",
#     "Samsng",
#     "Galxy",
#     "Samsung Glxy A14",
#     "iphne 12",
#     "Xiomi Redmi Nt 12",
#     "1+ Nord",
#     "Google Pixl 7",
#     "Huwei P50 Pro",
#     "iPhone 14 Pr Max",
#     "1+ 10 Pro",
#     "OnePlus 10T",
#     "Xiaomi Rdmi Note 11",
#     "Samsung Galxy S23 Ultr",
#     "iPhone 13 Pro Mx",
#     "Pixel 7 pr",
#     "1 + Nord CE",
#     "Samsung A54",
#     "Huawei P40 pr"
# ]
#
# # Run batch correction
# corrected_queries = corrector.refine_spelling_batch(test_queries)
#
# # Display results
# for original, corrected in zip(test_queries, corrected_queries):
#     print(f"Original: {original}")
#     print(f"Corrected: {corrected}")
#     print("-" * 40)