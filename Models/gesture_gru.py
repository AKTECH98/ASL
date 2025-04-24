import torch
import torch.nn as nn

class GestureGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=False,dropout=0):
        super(GestureGRU, self).__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths):
        # print("Forward input x:", x.shape)
        # print("Lengths:", lengths)
        # if (lengths == 0).any():
        #     raise ValueError(f"Zero-length sequence detected: {lengths}")
        # print(">> Packed Input Shape:", x.shape)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # print(">> Packed OK")
        _, hn = self.gru(packed)

        if self.bidirectional:
            hn = hn.view(self.num_layers, 2, x.size(0), self.hidden_size)
            hn = torch.cat((hn[-1, 0], hn[-1, 1]), dim=-1)
        else:
            hn = hn[-1]

        return self.fc(self.dropout(hn))
        # return self.fc(hn)
