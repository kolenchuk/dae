import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def text_to_indices(text, char_to_idx, add_eos: bool = False, max_length: int = 100):
    indices = []

    for char in text:
        if char.lower() in char_to_idx:
            indices.append(char_to_idx[char.lower()])
        else:
            logger.warning(f"Unknown character encountered: {char}")
            continue

    if add_eos:
        indices.append(char_to_idx['<EOS>'])

    if len(indices) > max_length:
        indices = indices[:max_length]
        if add_eos:
            indices[-1] = char_to_idx['<EOS>']

    return indices


def indices_to_text(indices, idx_to_char):
    special_tokens = {
        0, #<PAD>
        1, #<EOS>
    }
    #print(idx_to_char)
    return ''.join([idx_to_char.get(str(idx), '') for idx in indices if idx not in special_tokens])
    # return "".join([
    #     idx_to_char[idx]
    #     for idx in indices
    #     if idx not in special_tokens and idx in idx_to_char
    # ])
