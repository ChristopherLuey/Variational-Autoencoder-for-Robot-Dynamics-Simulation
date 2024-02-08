import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Activation after each layer except the last
                self.model.add_module(f"activation_{i}", activation())

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, output_activation=None):
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Activation after each layer except before the last
                self.model.add_module(f"activation_{i}", activation())
        if output_activation:  # Optional output activation function
            self.model.add_module("output_activation", output_activation())

    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, output_activation=None):
        super(Autoencoder, self).__init__()
        # Define encoder layer sizes including the input and latent layers
        full_encoder_sizes = [input_size] + encoder_layer_sizes + [latent_size]
        self.encoder = Encoder(full_encoder_sizes)

        # Define decoder layer sizes including the latent and output layers
        full_decoder_sizes = [latent_size] + decoder_layer_sizes + [input_size]
        self.decoder = Decoder(full_decoder_sizes, output_activation=output_activation)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss for non-binary data
        self.optimizer = None  # Optimizer will be defined in training method
        self.last_loss = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, input_batch, learning_rate=1e-3):
        self.train()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        output = self.forward(input_batch)
        loss = self.criterion(output, input_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()
        return self.last_loss

    def evaluate(self, input_batch):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_batch)
            loss = self.criterion(output, input_batch)
        return output, loss.item()

    @property
    def objective_function_value(self):
        return self.last_loss
