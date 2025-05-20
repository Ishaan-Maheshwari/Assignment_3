import torch

def encode(seq, stoi, add_special=False):
    ids = [stoi[c] for c in seq]
    if add_special:
        ids = [stoi["<sos>"]] + ids + [stoi["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)

def decode(indices, itos):
    return ''.join([itos[i] for i in indices if i >= 0])
