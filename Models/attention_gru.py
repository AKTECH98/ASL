import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, outputs, mask):
        # outputs: [batch, seq_len, hidden_size]
        energy = self.attn(outputs).squeeze(-1)  # [batch, seq_len]
        energy = energy.masked_fill(mask == 0, -1e9)  # Mask padding positions
        attn_weights = F.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # [batch, hidden_size]
        return context

class AttentionGestureGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=False, dropout=0):
        super(AttentionGestureGRU, self).__init__()

        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.attention = Attention(hidden_size, bidirectional)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Create mask
        mask = torch.arange(outputs.size(1), device=lengths.device)[None, :] < lengths[:, None]

        # Apply attention
        context = self.attention(outputs, mask)

        return self.fc(self.dropout(context))
