import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, batch_first=True, return_sequence=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequence = return_sequence

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        # If batch_first=True, x has shape [batch_size, seq_len, input_size]
        # If batch_first=False, x has shape [seq_len, batch_size, input_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.batch_first:
            last_time_step = lstm_out[:, -1, :]
        else:
            last_time_step = lstm_out.index_select(0, torch.arange(x.size(0) - 1, -1, -1).to(x.device))
            last_time_step = last_time_step.squeeze(0)
        if self.return_sequence:
            out = lstm_out
        else:
            out = self.fc(last_time_step)
        return out


class BiGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, batch_first=True, return_sequence=True):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequence = return_sequence

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        
        self.fc = nn.Linear(2 * hidden_size, output_size)

        self.input_dropout = nn.Dropout(0.25)
        self.output_dropout = nn.Dropout(0.5)

    def forward(self, x):
        # If batch_first=True, x has shape [batch_size, seq_len, input_size]
        # If batch_first=False, x has shape [seq_len, batch_size, input_size]
        gru_out, _ = self.gru(x)
        if self.batch_first:
            last_time_step = gru_out[:, -1, :]
        else:
            last_time_step = gru_out.index_select(0, torch.arange(x.size(0) - 1, -1, -1).to(x.device))
            last_time_step = last_time_step.squeeze(0)
        if self.return_sequence:
            out = gru_out
        else:
            out = self.fc(last_time_step)
        return out