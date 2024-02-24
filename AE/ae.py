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

class AugmentedAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, output_activation=None):
        super(AugmentedAutoencoder, self).__init__()
        # Define encoder layer sizes including the input and latent layers
        full_encoder_sizes = [input_size] + encoder_layer_sizes + [latent_size]
        self.encoder = Encoder(full_encoder_sizes)

        # Define decoder layer sizes
        full_decoder_sizes = [latent_size] + decoder_layer_sizes + [input_size]
        self.decoder = Decoder(full_decoder_sizes, output_activation=output_activation)

        # Define task network
        self.task_network = TaskNetwork(latent_size)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss for non-binary data
        self.last_loss = None
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):
        encoded = self.encoder(x)
        task_pred = self.task_network(encoded)
        decoded = self.decoder(encoded)
        return decoded, task_pred

    # def train_model(self, input_batch):
    #     self.train()
    #     output = self.forward_autoencoder(input_batch)
    #     loss = self.criterion(output, input_batch)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.last_loss = loss.item()
    #     return self.last_loss
    
    def train_model(self, input_batch, target_value):
        self.train()
        
        # Forward pass through the autoencoder
        decoded, task_pred = self.forward(input_batch)

        # Calculate the reconstruction loss
        reconstruction_loss = self.criterion(decoded, input_batch)
                
        # Calculate the task-specific loss
        task_loss = -(torch.sum(input_batch, dim=1) - target_value) ** 2
        #task_loss = self.criterion(task_pred, target_value)
        
        # Combine the losses
        combined_loss = reconstruction_loss + task_loss
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Backward pass and optimize
        combined_loss.backward()
        self.optimizer.step()
        
        self.last_loss = combined_loss.item()
        return (reconstruction_loss.item(), task_loss.item(), self.last_loss)

    def evaluate(self, input_batch, target_value):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(input_batch)
            decoded = self.decoder(encoded)
        
            # Calculate the reconstruction loss
            reconstruction_loss = self.criterion(decoded, input_batch)
        
            # Forward pass through the task network
            task_pred = self.task_network(encoded)
            
            #task_loss = self.criterion(task_pred, target_value)
            task_loss = (torch.sum(input_batch, dim=1) - target_value) ** 2

        return (decoded, task_pred), reconstruction_loss+task_loss.item()

    @property
    def objective_function_value(self):
        return self.last_loss
    

class TaskNetwork(nn.Module):
    def __init__(self, latent_size):
        super(TaskNetwork, self).__init__()
        self.fc = nn.Linear(latent_size, 1)

    def forward(self, x):
        return self.fc(x)
