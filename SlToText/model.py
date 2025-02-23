import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_out = self.fc(lstm_out[:, -1, :])
        return final_out
