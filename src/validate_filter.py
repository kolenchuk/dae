import json

with open('validation_results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

filtered_data = [
    record for record in data
    if record['levenshtein_distance'] != 0 or record['bleu_score'] != 100.0
]

print(json.dumps(filtered_data, indent=4, ensure_ascii=False))

with open('filtered_validation_results.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, indent=4, ensure_ascii=False)
