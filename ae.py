import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_size, num_layers, input_size)

    def forward(self, x):
        latent = self.encoder(x)
        # Replicate latent code across time steps
        latent = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        return self.decoder(latent)

