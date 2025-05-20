import torch
from torch.utils.data import Dataset
from vocab import encode
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.nn.utils.rnn import pad_sequence

class TransliterationDataset(Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi):
        self.pairs = pairs
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # src, tgt = self.pairs[idx]
        tgt, src = self.pairs[idx]
        src_ids = encode(src, self.src_stoi)
        tgt_ids = encode(tgt, self.tgt_stoi)
        return src_ids, tgt_ids

def collate_fn(batch):
    src_seqs = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    tgt_seqs = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    
    src_pad = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_pad, tgt_pad
