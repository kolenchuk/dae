from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import os
import json

MODEL_DIR = "models/bart-base"
os.makedirs(MODEL_DIR, exist_ok=True)

if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    print(f"BART model already downloaded at {MODEL_DIR}")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    for key in ["num_beams", "early_stopping", "no_repeat_ngram_size", "forced_bos_token_id"]:
        if hasattr(model.config, key):
            delattr(model.config, key)

    gen_config_dict = {
        "num_beams": 4,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "forced_bos_token_id": 0
    }

    with open(os.path.join(MODEL_DIR, "generation_config.json"), "w") as f:
        json.dump(gen_config_dict, f)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"BART model downloaded and saved at {MODEL_DIR}")

