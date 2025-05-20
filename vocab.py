import torch

def encode(seq, stoi):
    unk_idx = stoi.get("<unk>", 3)  # default index if <unk> is missing
    ids = [stoi.get(c, unk_idx) for c in seq]
    return [stoi["<sos>"]] + ids + [stoi["<eos>"]]


def decode(indices, itos):
    return ''.join([itos[i] for i in indices if i >= 0])
