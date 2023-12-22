from ae import *

input_size = 32   # Size of the input nodes
hidden_size = 64  # Hidden layer size
num_layers = 2    # Number of LSTM layers
latent_dim = 16   # Size of the latent space

autoencoder = LSTMAutoencoder(input_size, hidden_size, num_layers, latent_dim)

example_data = torch.randn(5, 10, input_size)
output = autoencoder(example_data)

print(output.shape)


