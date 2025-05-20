import torch
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from tabulate import tabulate

from data_utils import read_tsv
from dataset import TransliterationDataset, collate_fn
from model import Seq2Seq

# === Config ===
TEST_PATH = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
MODEL_PATH = "checkpoints/best_model.pth"

def load_vocab(json_path):
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)

def decode_sequence(seq, itos):
    result = []
    for idx in seq:
        if 0 <= idx < len(itos):
            ch = itos[idx]
            if ch == "<eos>":
                break
            if ch not in ["<sos>", "<pad>"]:
                result.append(ch)
    return ''.join(result)

def evaluate_test():
    # === Load vocabs ===
    src_stoi = load_vocab("src_stoi.json")
    tgt_itos = load_vocab("tgt_itos.json")  # this is a list
    tgt_stoi = {char: idx for idx, char in enumerate(tgt_itos)}

    # Build src_itos from src_stoi
    src_itos = [None] * len(src_stoi)
    for ch, idx in src_stoi.items():
        src_itos[idx] = ch

    # === Load test data ===
    test_pairs = read_tsv(TEST_PATH)
    test_ds = TransliterationDataset(test_pairs, src_stoi, tgt_stoi)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # === Load model ===
    model = Seq2Seq(
        input_vocab_size=len(src_stoi),
        target_vocab_size=len(tgt_itos),
        embedding_dim=256,
        hidden_dim=256,
        num_layers=3,
        cell_type="LSTM",
        dropout=0.2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # === Predict & Evaluate ===
    exact_matches = 0
    total = 0
    predictions = []

    os.makedirs("predictions_vanilla", exist_ok=True)
    output_path = "predictions_vanilla/test_predictions.tsv"

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("Input\tTarget\tPrediction\n")

        for src, tgt in tqdm(test_dl):
            src, tgt = src.to(device), tgt.to(device)

            with torch.no_grad():
                output = model(src, tgt=None, teacher_forcing_ratio=0)

            pred_idxs = output.argmax(dim=-1).squeeze(0).tolist()
            tgt_idxs = tgt.squeeze(0).tolist()
            src_idxs = src.squeeze(0).tolist()

            pred_str = decode_sequence(pred_idxs, tgt_itos)
            tgt_str = decode_sequence(tgt_idxs, tgt_itos)
            src_str = decode_sequence(src_idxs, src_itos)

            if pred_str == tgt_str:
                exact_matches += 1

            predictions.append((src_str, tgt_str, pred_str))
            out_f.write(f"{src_str}\t{tgt_str}\t{pred_str}\n")

            total += 1

    accuracy = exact_matches / total
    print(f"\n✅ Exact Match Accuracy on Test Set: {accuracy:.4f} ({exact_matches}/{total})")

    # === Display sample predictions ===
    print("\n📌 Sample Predictions:")
    print(tabulate(predictions[:10], headers=["Input (Latin)", "Target (Devanagari)", "Prediction"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    evaluate_test()
