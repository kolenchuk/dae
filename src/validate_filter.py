import json

# Завантаження даних з JSON файлу
with open('validation_results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Фільтрація записів
filtered_data = [
    record for record in data
    if record['levenshtein_distance'] != 0 or record['bleu_score'] != 100.0
]

# Вивід результатів
print(json.dumps(filtered_data, indent=4, ensure_ascii=False))

# Збереження результатів у новий JSON файл
with open('filtered_validation_results.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, indent=4, ensure_ascii=False)
