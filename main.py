import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import read_tsv, build_vocab
from dataset import TransliterationDataset, collate_fn
from model import Seq2Seq
from train import train_one_epoch, calculate_accuracy

def evaluate(model, dataloader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            acc = calculate_accuracy(output[:, 1:], tgt[:, 1:], pad_idx)

            total_loss += loss.item()
            total_acc += acc
            count += 1

    return total_loss / count, total_acc / count


def train():
    wandb.init()
    config = wandb.config

    # Load data
    train_pairs = read_tsv(config.train_path)
    dev_pairs = read_tsv(config.dev_path)

    src_stoi, src_itos = build_vocab([x[0] for x in train_pairs])
    tgt_stoi, tgt_itos = build_vocab([x[1] for x in train_pairs])

    train_ds = TransliterationDataset(train_pairs, src_stoi, tgt_stoi)
    dev_ds = TransliterationDataset(dev_pairs, src_stoi, tgt_stoi)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(
        input_vocab_size=len(src_stoi),
        target_vocab_size=len(tgt_stoi),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        cell_type=config.cell_type,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_stoi["<pad>"])

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, tgt_stoi["<pad>"], device)
        val_loss, val_acc = evaluate(model, dev_dl, criterion, tgt_stoi["<pad>"], device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Optionally save model checkpoint at end
    if wandb.run is not None:
        torch.save(model.state_dict(), "model.pth")
        wandb.save("model.pth")
    
    import json
    with open('src_stoi.json', 'w', encoding='utf-8') as f:
        json.dump(src_stoi, f, ensure_ascii=False)

    with open('tgt_itos.json', 'w', encoding='utf-8') as f:
        json.dump(tgt_itos, f, ensure_ascii=False)


if __name__ == "__main__":
    train()
