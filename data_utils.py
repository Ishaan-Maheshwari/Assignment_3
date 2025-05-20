import os
from collections import Counter

def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            data.append(parts[:2])
    return data


def build_vocab(sequences, special_tokens=["<pad>", "<sos>", "<eos>"]):
    chars = [ch for seq in sequences for ch in seq]
    counter = Counter(chars)
    itos = special_tokens + sorted(counter)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos
