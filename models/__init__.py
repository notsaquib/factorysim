import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, n_actions=1):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_actions + 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetworkLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, lstm_layers=1, n_actions=1):
        super(QNetworkLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, n_actions + 1)

    def forward(self, x):
        device = self.device
        batch_size = x.size(0)
        if batch_size == 1:
            h0 = torch.zeros(self.lstm_layers, self.hidden_size).to(device)
            c0 = torch.zeros(self.lstm_layers, self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        x, _ = self.lstm(x, (h0, c0))
        if batch_size ==1:
            x = torch.tanh(self.fc1(x[:, :]))
        else:
            x = torch.tanh(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x

