import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, embedding_dim, hidden_dim, num_layers=1, cell_type="LSTM", dropout=0.0):
        super().__init__()
        self.cell_type = cell_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.decoder_dropout = nn.Dropout(dropout)

        rnn_class = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.encoder = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_len, self.fc_out.out_features).to(src.device)

        embedded_src = self.encoder_embedding(src)
        embedded_src = self.encoder_dropout(embedded_src)
        encoder_outputs, hidden = self.encoder(embedded_src)

        if self.cell_type == "LSTM":
            h, c = hidden
            decoder_hidden = (h.detach(), c.detach())
        else:
            decoder_hidden = hidden.detach()

        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            embedded = self.decoder_embedding(input_token)
            embedded = self.decoder_dropout(embedded)
            embedded = embedded.unsqueeze(1)  # (batch, 1, embed_dim)

            decoder_output, decoder_hidden = self.decoder(embedded, decoder_hidden)
            logits = self.fc_out(decoder_output.squeeze(1))
            outputs[:, t] = logits

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else logits.argmax(1)

        return outputs

    def beam_search_decode(self, src, beam_width=3, max_len=30, sos_idx=1, eos_idx=2, device='cpu'):
        """
        Perform beam search decoding for a batch of src sequences.
        """
        # For brevity, this example shows single sentence beam search.
        # You can extend it to batch with careful bookkeeping.

        self.eval()
        with torch.no_grad():
            embedded_src = self.encoder_embedding(src)
            embedded_src = self.encoder_dropout(embedded_src)
            encoder_outputs, hidden = self.encoder(embedded_src)

            if self.cell_type == "LSTM":
                h, c = hidden
                decoder_hidden = (h, c)
            else:
                decoder_hidden = hidden

            # Initialize beams with (tokens, score, hidden)
            beams = [([sos_idx], 0.0, decoder_hidden)]

            for _ in range(max_len):
                new_beams = []
                for tokens, score, hidden in beams:
                    if tokens[-1] == eos_idx:
                        # Beam already ended
                        new_beams.append((tokens, score, hidden))
                        continue

                    input_token = torch.tensor([tokens[-1]], device=device)
                    embedded = self.decoder_embedding(input_token).unsqueeze(1)
                    embedded = self.decoder_dropout(embedded)
                    decoder_output, hidden_next = self.decoder(embedded, hidden)
                    logits = self.fc_out(decoder_output.squeeze(1))
                    log_probs = torch.log_softmax(logits, dim=1)

                    top_log_probs, top_indices = log_probs.topk(beam_width)

                    for log_p, idx in zip(top_log_probs[0], top_indices[0]):
                        new_seq = tokens + [idx.item()]
                        new_score = score + log_p.item()
                        new_beams.append((new_seq, new_score, hidden_next))

                # Keep top beam_width beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            best_seq = beams[0][0]
            return best_seq
