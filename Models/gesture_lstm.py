import torch.nn as nn
import torch
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, bidirectional=False):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)
        # Use last hidden state
        if self.lstm.bidirectional:
            hn = hn.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
            hn = torch.cat((hn[-1, 0], hn[-1, 1]), dim=-1)
        else:
            hn = hn[-1]
        out = self.fc(hn)
        return out
