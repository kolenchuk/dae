from fastapi import FastAPI
import torch

from src.dae import DAE
from src.mobile_query_corrector import MobileQueryCorrector
import os
import pandas as pd
from difflib import get_close_matches
from Levenshtein import distance
from transformers import pipeline
import json
import logging
from src.text_utils import text_to_indices, indices_to_text

from src.model_save_utils import load_checkpoint_distributed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_smartphone_models(filepath):
    """ Loads the list of available smartphone models """
    df = pd.read_csv(filepath)
    return df['model'].str.lower().tolist()

with open("./data/unique_words.txt", "r", encoding="utf-8") as f:
    DICTIONARY = set(word.strip().lower() for word in f.readlines())

SMARTPHONES = load_smartphone_models("./data/smartphones.csv")

app = FastAPI()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./models/dae_best_checkpoint"
MODEL_CLASS = DAE
model, metadata = load_checkpoint_distributed(MODEL_PATH, MODEL_CLASS, DEVICE)
CHAR_TO_IDX = metadata['char_to_idx']
IDX_TO_CHAR = metadata['idx_to_char']
model.eval()

BART_MODEL_DIR = "./models/bart-base"

gen_config_path = os.path.join(BART_MODEL_DIR, "generation_config.json")
with open(gen_config_path, "r") as f:
    gen_config_dict = json.load(f)

BART_TEXT_CORRECTOR = pipeline(
    "text2text-generation",
    model=BART_MODEL_DIR,
    tokenizer=BART_MODEL_DIR,
    device=DEVICE
)

MOBILE_QUERY_CORRECTOR = MobileQueryCorrector(BART_TEXT_CORRECTOR)


def correct_word(word: str) -> str:
    """
    Виправляє слово за допомогою моделі, якщо воно відсутнє в словнику.
    Якщо виправлення не вдалось – повертає оригінальне слово.
    """
    if word in DICTIONARY or len(word) < 3 or any(c.isdigit() for c in word):
        return word  # Не змінюємо
    else:
        with torch.no_grad():
            indices = text_to_indices(word, CHAR_TO_IDX)
            input_tensor = torch.tensor([indices], device=DEVICE)
            output_tensor = model(input_tensor)
            predictions = output_tensor.argmax(dim=-1)
            corrected_word = indices_to_text(predictions[0].tolist(), IDX_TO_CHAR)

        return corrected_word if corrected_word in DICTIONARY else ""


def get_suggestions(user_query, n=5, threshold=0.6):
    """ Returns a list of possible matches for the user query """
    user_query = user_query.lower()

    substring_matches = [model for model in SMARTPHONES if user_query in model]

    if substring_matches:
        return substring_matches[:n]

    close_matches = get_close_matches(user_query, SMARTPHONES, n=n, cutoff=threshold)

    lev_matches = sorted(SMARTPHONES, key=lambda x: distance(user_query, x))[:n]

    suggestions = list(dict.fromkeys(substring_matches + close_matches + lev_matches))
    return suggestions[:n]

@app.get("/correct_dae")
def correct_query_with_dae(query: str):
    """
    API-ендпоінт для виправлення пошукового запиту.
    """
    words = query.lower().strip().split()
    corrected_words = []
    has_uncorrected = False

    for word in words:
        corrected = correct_word(word)
        if not corrected:
            corrected = word
            has_uncorrected = True

        corrected_words.append(corrected)

    corrected_query = " ".join(corrected_words)

    return {"original": query, "corrected": corrected_query, "has_uncorrected": has_uncorrected}

@app.get("/correct_bart")
def correct_query_with_bart(query: str):
    corrected_query = MOBILE_QUERY_CORRECTOR.refine_spelling(query)

    return {"original": query, "corrected": corrected_query}


@app.get("/search")
def search(query: str):
    """
    API-ендпоінт для пошуку моделей телефонів.
    """
    logger.info(f"Received search query: {query}")
    corrected_query = correct_query_with_dae(query)
    logger.info(f"Corrected query with dae: {corrected_query}")
    if corrected_query["has_uncorrected"]:
        corrected_query = correct_query_with_bart(corrected_query["corrected"])
        logger.info(f"Corrected query with bart: {corrected_query}")

    suggestions = get_suggestions(corrected_query["corrected"])

    return {"original": query, "search_variants": suggestions}

@app.get("/")
def read_root():
    return {"message": "API is working"}