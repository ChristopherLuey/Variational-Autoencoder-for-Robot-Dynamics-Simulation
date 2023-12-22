import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dim):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = np.linspace(input_size, latent_dim, num_layers)

        for i in range(num_layers - 1):
            self.layers.append(nn.LSTM(int(layer_sizes[i]), int(layer_sizes[i+1]), batch_first=True))

        self.fc = nn.Linear(int(layer_sizes[-1]), latent_dim)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        x = x[:, -1, :]  # Taking the last time step
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = np.linspace(latent_dim, input_size, num_layers)

        for i in range(num_layers - 1):
            self.layers.append(nn.LSTM(int(layer_sizes[i]), int(layer_sizes[i+1]), batch_first=True))

        self.fc = nn.Linear(int(layer_sizes[-1]), input_size)

    def forward(self, x, sequence_length):
        x = x.unsqueeze(1).repeat(1, sequence_length, 1)  # Repeat latent vector
        for layer in self.layers:
            x, _ = layer(x)
        return self.fc(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dim, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, num_layers, latent_dim)
        self.decoder = Decoder(input_size, latent_dim, num_layers)
        self.sequence_length = sequence_length

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent, self.sequence_length)
