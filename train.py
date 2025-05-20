import torch
from tqdm import tqdm

def calculate_accuracy(pred, target, pad_idx):
    pred_classes = pred.argmax(dim=-1)
    mask = target != pad_idx
    correct = (pred_classes == target) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def train_one_epoch(model, dataloader, optimizer, criterion, pad_idx, device):
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, tgt[:, 1:], pad_idx)

        total_loss += loss.item()
        total_acc += acc
        count += 1

    return total_loss / count, total_acc / count

