import pandas as pd

unique_words_path = '../data/unique_words.txt'
data_phones_path = '../data/data_phones_unique_words.csv'

with open(unique_words_path, 'r') as file:
    unique_words = set(file.read().splitlines())

data_phones = pd.read_csv(data_phones_path)

phone_words = set(data_phones['word'].str.lower().unique())

new_words = phone_words - unique_words

with open(unique_words_path, 'a') as file:
    for word in sorted(new_words):
        file.write(f"{word}\n")