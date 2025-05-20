import os
from collections import Counter

def read_tsv(filename):
    with open(filename, encoding='utf-8') as f:
        pairs = []
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                tgt, src = parts[0], parts[1]  # tgt = Devanagari, src = Latin
                pairs.append((src, tgt))
    return pairs



def build_vocab(sequences):
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab = set(char for seq in sequences for char in seq)
    itos = special_tokens + sorted(vocab)
    stoi = {char: i for i, char in enumerate(itos)}
    return stoi, itos

