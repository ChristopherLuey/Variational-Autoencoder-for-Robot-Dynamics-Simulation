import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CVAEEncoder(nn.Module):
    def __init__(self, layer_sizes, latent_space, activation=nn.ReLU):
        super(CVAEEncoder, self).__init__()
        self.model = nn.Sequential()

        for i in range(len(layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.model.add_module(f"activation_{i}", activation())

        self.mean = nn.Linear(layer_sizes[-1], latent_space)
        self.log_variation = nn.Linear(layer_sizes[-1], latent_space)

    def forward(self, x):
        z = self.model(x)
        mean = self.mean(z)
        log_variation = self.log_variation(z)
        return mean, log_variation

class CVAEDecoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, activation=nn.ReLU, output_activation=None):
        super(CVAEDecoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1])) 
            # if i < len(layer_sizes):
            self.model.add_module(f"activation_{i}", activation())

        if output_activation:  # Optional output activation function
            self.model.add_module("output_activation", output_activation())

    def forward(self, x):
        return self.model(x)

class AugmentedConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, task_layer_sizes, condition_size, output_activation=None):
        super(AugmentedConditionalVariationalAutoencoder, self).__init__()
        # Define encoder layer sizes including the input and latent layers
        full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)

        # Define decoder layer sizes
        full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)

        # Define task network
        self.task_network = TaskNetwork(latent_size, task_layer_sizes)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss for non-binary data
        self.last_loss = None
        self.learning_rate = 1e-1
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


    def reparameterize(self, mean, log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        return mean + eps * std


    def forward(self, x, condition):
        x=x[0]
        # print(torch.cat((x, condition), dim=-1))
        mean, log_variation = self.encoder(torch.cat((x, condition), dim=-1))
        latent_representation = self.reparameterize(mean, log_variation)
        task_pred = self.task_network(mean)
        decoded = self.decoder(torch.cat((latent_representation, condition), dim=-1))
        return decoded, task_pred, latent_representation, mean, log_variation

    # def train_model(self, input_batch):
    #     self.train()
    #     output = self.forward_autoencoder(input_batch)
    #     loss = self.criterion(output, input_batch)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.last_loss = loss.item()
    #     return self.last_loss

    def loss_function(self,input_batch, decoded, mean, log_variation):
        reproduction_loss = self.criterion(decoded, input_batch)
        # reproduction_loss = nn.functional.mse_loss(decoded, input_batch[0], reduction='sum')
        KLD = -0.0 * torch.sum(1+ log_variation - mean.pow(2) - log_variation.exp())
        return reproduction_loss + KLD
    
    def train_model(self, input_batch, target_value, condition):
        self.train()
        
        # Forward pass through the autoencoder
        decoded, task_pred, encoded, mean, log_variation = self.forward(input_batch, condition)

        # Calculate the reconstruction loss
        reconstruction_loss = self.loss_function(input_batch, decoded, mean, log_variation)
                
        # Calculate the task-specific loss
        # task_loss = (torch.sum(input_batch, dim=1) - target_value) ** 2
        task_loss = self.criterion(task_pred, target_value)
        
        # Combine the losses
        combined_loss = 1*reconstruction_loss + task_loss
        
        # Zero the gradientss
        self.optimizer.zero_grad()
        
        # Backward pass and optimize
        combined_loss.backward()
        self.optimizer.step()
        
        self.last_loss = combined_loss.item()
        return (reconstruction_loss.item(), task_loss.item(), self.last_loss, encoded)

    def evaluate(self, input_batch, target_value, condition):
        self.eval()
        with torch.no_grad():
            decoded, task_pred, encoded, mean, log_variation = self.forward(input_batch, condition)
        
            # Calculate the reconstruction loss
            reconstruction_loss = self.loss_function(input_batch, decoded, mean, log_variation)
        
            # Forward pass through the task network
            task_pred = self.task_network(encoded)
            
            # task_loss = self.criterion(task_pred, target_value)
            task_loss = -task_pred**2
            # task_loss = (torch.sum(input_batch, dim=1) - target_value) ** 2
            combined_loss = 1*reconstruction_loss + task_loss

        return (decoded, task_pred), combined_loss
    

    def evaluate_gradient(self, input_batch, target_value, condition):
        """
        Evaluate the model to compute the loss for the given input batch and target value.
        
        Args:
        - input_batch (torch.Tensor): The input batch.
        - target_value (torch.Tensor): The target value for the task network.
        
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the decoded output and the task prediction.
        - torch.Tensor: The combined loss (reconstruction loss + task loss).
        """
        self.eval()
        # Ensure target_value has the correct shape for criterion comparison
        target_value = target_value.unsqueeze(1) if target_value.dim() == 1 else target_value
        
        decoded, task_pred, encoded, mean, log_variation = self.forward(input_batch, condition)
    
        # Calculate the reconstruction loss
        reconstruction_loss = self.loss_function(input_batch, decoded, mean, log_variation)
    
        # Forward pass through the task network
        task_pred = self.task_network(encoded)
        
        # task_loss = self.criterion(task_pred, target_value)
        if task_pred < 0:

            task_loss = task_pred**2
        else:
            task_loss = -task_pred**2

        # task_loss = (torch.sum(input_batch, dim=1) - target_value) ** 2
        # Combine the losses
        combined_loss = 1*reconstruction_loss + task_loss

        # Return both decoded output and task prediction, and the combined loss
        return (decoded, task_pred), combined_loss

    @property
    def objective_function_value(self):
        return self.last_loss
    

class TaskNetwork(nn.Module):
    def __init__(self, latent_size, task_layer_sizes, activation=nn.ReLU):
        super(TaskNetwork, self).__init__()
        self.model = nn.Sequential()
        # self.fc = nn.Linear(latent_size, 1)

        for i in range(len(task_layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(task_layer_sizes[i], task_layer_sizes[i+1]))
            if i < len(task_layer_sizes) - 2:  # Activation after each layer except the last
                self.model.add_module(f"activation_{i}", activation())

    def forward(self, x):
        return self.model(x)


