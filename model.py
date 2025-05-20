import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, embedding_dim, hidden_dim, num_layers=1, cell_type="LSTM", dropout=0.0, use_attention=True):
        super().__init__()
        self.cell_type = cell_type
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(dropout)

        rnn_class = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]

        self.encoder = rnn_class(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = rnn_class(embedding_dim + hidden_dim if use_attention else embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.attention = Attention(hidden_dim) if use_attention else None
        self.fc_out = nn.Linear(hidden_dim * 2 if use_attention else hidden_dim, target_vocab_size)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else 30
        outputs = torch.zeros(batch_size, tgt_len, self.fc_out.out_features).to(src.device)

        # Encode
        embedded_src = self.encoder_dropout(self.encoder_embedding(src))
        encoder_outputs, hidden = self.encoder(embedded_src)

        if self.cell_type == "LSTM":
            h, c = hidden
            decoder_hidden = (h.detach(), c.detach())
        else:
            decoder_hidden = hidden.detach()

        input_token = tgt[:, 0] if tgt is not None else torch.full((batch_size,), 1, dtype=torch.long).to(src.device)

        for t in range(1, tgt_len):
            embedded = self.decoder_dropout(self.decoder_embedding(input_token)).unsqueeze(1)

            if self.use_attention:
                if self.cell_type == "LSTM":
                    decoder_hidden_last = decoder_hidden[0][-1]
                else:
                    decoder_hidden_last = decoder_hidden[-1]

                attn_weights = self.attention(decoder_hidden_last, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
                rnn_input = torch.cat((embedded, context), dim=2)
            else:
                rnn_input = embedded

            decoder_output, decoder_hidden = self.decoder(rnn_input, decoder_hidden)
            output_logits = self.fc_out(torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)) if self.use_attention else self.fc_out(decoder_output.squeeze(1))
            outputs[:, t] = output_logits

            if tgt is not None:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = tgt[:, t] if teacher_force else output_logits.argmax(1)
            else:
                input_token = output_logits.argmax(1)

        return outputs
