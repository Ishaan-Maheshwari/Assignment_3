import torch
from model import Seq2Seq
from data_utils import build_vocab
import argparse

def encode_input(text, stoi):
    # Encode input string to list of token indices
    return [stoi.get(c, stoi["<unk>"]) for c in text]

def decode_output(indices, itos):
    # Convert list of token indices back to string
    tokens = [itos[i] for i in indices]
    # Remove tokens after <eos>
    if "<eos>" in tokens:
        tokens = tokens[:tokens.index("<eos>")]
    return "".join(tokens)

def load_model(model_path, input_vocab_size, target_vocab_size, embedding_dim, hidden_dim, num_layers, cell_type, dropout, device):
    model = Seq2Seq(
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cell_type=cell_type,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabularies (you must save these after training or rebuild from train data)
    # Here, we assume you saved the vocab dicts somewhere and load them.
    import json
    with open(args.src_stoi_path, 'r', encoding='utf-8') as f:
        src_stoi = json.load(f)
    with open(args.tgt_itos_path, 'r', encoding='utf-8') as f:
        tgt_itos = json.load(f)

    # Convert tgt_itos (list) to dict if needed for decoding
    if isinstance(tgt_itos, dict):
        tgt_itos = {int(k): v for k, v in tgt_itos.items()}

    # Inverse of src_stoi might not be needed here, but just for completeness:
    src_vocab_size = len(src_stoi)
    tgt_vocab_size = len(tgt_itos)

    model = load_model(
        args.model_path,
        input_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        dropout=args.dropout,
        device=device
    )

    sos_idx = src_stoi.get("<sos>", 1)
    eos_idx = src_stoi.get("<eos>", 2)

    # Encode input text
    input_seq = encode_input(args.input_text, src_stoi)
    input_tensor = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)  # batch=1

    # Beam search decoding
    beam_output = model.beam_search_decode(
        input_tensor,
        beam_width=args.beam_width,
        max_len=args.max_len,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        device=device
    )

    # Decode output tokens to string
    transliteration = decode_output(beam_output, tgt_itos)
    print(f"Input: {args.input_text}")
    print(f"Transliteration: {transliteration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq Beam Search Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--src_stoi_path", type=str, required=True, help="Path to source stoi JSON")
    parser.add_argument("--tgt_itos_path", type=str, required=True, help="Path to target itos JSON")
    parser.add_argument("--input_text", type=str, required=True, help="Input text to transliterate")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["RNN", "GRU", "LSTM"], help="RNN cell type")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam search width")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum output length")
    args = parser.parse_args()

    main(args)

    print("Inference completed.")