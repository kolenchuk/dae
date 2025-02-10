import pandas as pd
import re

file_path = './data/data_phones.csv'
data = pd.read_csv(file_path)

data['title'] = data['title'].apply(lambda x: re.sub(r'[^A-Za-z0-9]', ' ', str(x)))

unique_words = set()
for title in data['title']:
    words = title.split()
    unique_words.update(words)

filtered_words = [word.lower() for word in unique_words if word.isalpha() and len(word) >= 4]

output_path = './data/data_phones_unique_words.csv'
pd.DataFrame(filtered_words, columns=['word']).to_csv(output_path, index=False)
