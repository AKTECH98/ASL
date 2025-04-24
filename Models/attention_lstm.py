import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=True, dropout=0):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        self.hidden_size = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)

        # Attention
        self.attn_fc = nn.Linear(self.hidden_size, 1)

        # Classification
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.dropout(out)  # Apply dropout after LSTM

        # Attention weights: (batch_size, seq_len, 1)
        attn_scores = self.attn_fc(out).squeeze(-1)  # (batch_size, seq_len)

        # Masking padded values
        mask = torch.arange(out.size(1)).unsqueeze(0).to(x.device) < lengths.unsqueeze(1)
        attn_scores[~mask] = float('-inf')  # set attention scores of pads to -inf

        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        context = torch.sum(out * attn_weights, dim=1)  # (batch_size, hidden)

        context = self.dropout(context)  # Apply dropout before classifier

        return self.classifier(context)
