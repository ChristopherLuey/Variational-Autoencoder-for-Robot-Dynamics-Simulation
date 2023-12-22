from ae import *
import torch

input_size = 32   # Size of the input nodes
num_layers = 4    # Number of LSTM layers
latent_dim = 16   # Size of the latent space
sequence_length = 10  # Length of the sequence

autoencoder = LSTMAutoencoder(input_size, num_layers, latent_dim, sequence_length)

example_input = torch.randn(5, sequence_length, input_size)
output = autoencoder(example_input)
print(output.shape)  # Check output shape
