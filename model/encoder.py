import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Transfomer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, nlayers, dropout=0.5):
        super(Transfomer_Encoder, self).__init__()

        self.input_size = input_size

        nhead = 1
        for h in range(5, 15):
            if self.input_size % h == 0:
                nhead = h
                break

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.lin_out = torch.nn.Linear(
                input_size, hidden_dim, bias=True
            )
    
    def forward(self, x, lengths):
        x = self.transformer_encoder(x)
        x = self.lin_out(x)
        return x


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, nlayers, dropout=0.5):
        super(LSTM_Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size,
                            hidden_dim // 2,
                            dropout=dropout,
                            bidirectional=True,
                            num_layers=nlayers,
                            batch_first=True)
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)

        packed_out, (_, _) = self.lstm(packed, None)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return out
