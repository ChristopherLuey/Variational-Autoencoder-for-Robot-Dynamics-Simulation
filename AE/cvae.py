import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

# from torchviz import make_dot
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
            if i < len(layer_sizes) - 2: 
                self.model.add_module(f"activation_{i}", activation())
        if output_activation is not None:
            self.model.add_module("output_activation", output_activation)

    def forward(self, x):
        return self.model(x)


class BasicAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, condition_size, output_activation=None):
        super(BasicAutoencoder, self).__init__()
        full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)
        full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)
        self.learning_rate = 1e-1
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x, condition):
        mean, log_variation = self.encoder(torch.cat((x, condition), dim=-1))
        latent_representation = self.reparameterize(mean, log_variation)
        decoded = self.decoder(torch.cat((latent_representation, condition), dim=-1))
        return decoded, latent_representation, mean, log_variation

    def reparameterize(self, mean, log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        return mean + eps * std
        # return mean

    def reparameterize_evaluate(self,mean,log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        # return mean + eps * std
        return mean

    def loss_function(self, input_batch, decoded, mean, log_variation):
        reproduction_loss = self.criterion(decoded, input_batch)
        # reproduction_loss = nn.functional.mse_loss(decoded, input_batch[0], reduction='sum')
        KLD = -0.0 * torch.sum(1+ log_variation - mean.pow(2) - log_variation.exp())
        return reproduction_loss + KLD

    def train_model(self, input_batch, condition):
        self.train()
        decoded, latent_representation, mean, log_variation= self.forward(input_batch, condition)
        loss = self.loss_function(input_batch, decoded, mean, log_variation)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), decoded, latent_representation, mean, log_variation

    def evaluate(self,input_batch, condition):
        self.eval()
        decoded, latent_representation, mean, log_variation = self.forward(input_batch, condition)
        loss = self.loss_function(input_batch, decoded, mean, log_variation)
        return loss.item(), decoded, latent_representation, mean, log_variation


class AugmentedConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, task_layer_sizes, condition_size, output_activation=None):
        super(AugmentedConditionalVariationalAutoencoder, self).__init__()
        # Define encoder layer sizes including the input and latent layers
        # full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        # self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)

        # # Define decoder layer sizes
        # full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        # self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)

        # Define task network

        self.autoencoder = BasicAutoencoder(input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, condition_size)

        task_layer_sizes[0] = task_layer_sizes[0] + condition_size
        self.task_network = TaskNetwork(latent_size, task_layer_sizes)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss for non-binary data
        self.last_loss = None
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer_autoencoder = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        self.optimizer_task_network = optim.Adam(self.task_network.parameters(), lr=1e-2)

        self.reconstruction_weight = 0.1
        self.task_weight = 1

        self.direction = torch.tensor([0.0], dtype=torch.float32, device="cuda:0")

    def reparameterize(self, mean, log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        return mean + eps * std
        # return mean

    def reparameterize_evaluate(self,mean,log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        # return mean + eps * std
        return mean

    def forward(self, x, condition, evaluating):
        decoded, latent_representation, mean, log_variation = self.autoencoder(x, condition)
        task_pred = self.task_network(latent_representation)
        return decoded, task_pred, latent_representation, mean, log_variation

    def train_model(self, input_batch, target_value, condition, _latent_representation):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.task_network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.autoencoder.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.autoencoder.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        print(self.task_network.parameters())


        self.train()
        self.autoencoder.encoder.train()
        self.autoencoder.decoder.train()
        self.task_network.train()
        # Zero the gradientss
        self.optimizer.zero_grad()

        # Forward pass through the autoencoder
        decoded, latent_representation, mean, log_variation = self.autoencoder.forward(input_batch, condition)

        # Calculate the reconstruction loss
        reconstruction_loss = self.autoencoder.loss_function(input_batch, decoded, mean, log_variation)
        # if _latent_representation.shape[0] == 0:
        #     _latent_representation = latent_representation
        # else:
        #     _latent_representation = torch.cat([_latent_representation, latent_representation[-1]])
        # Calculate the task-specific loss
        # task_pred = self.task_network.forward(_latent_representation, condition)

        if _latent_representation.shape[0] == 0:
            _latent_representation = mean
        else:
            _latent_representation = torch.cat([_latent_representation, mean[-1]])

        task_pred = self.task_network.forward(_latent_representation, condition)
        task_loss = self.task_network.criterion(task_pred, target_value)

        # Combine the losses
        combined_loss = reconstruction_loss * self.reconstruction_weight + task_loss * self.task_weight

        # Backward pass and optimize
        combined_loss.backward()
        self.optimizer.step()

        self.last_loss = combined_loss.item()
        return (reconstruction_loss, task_loss, reconstruction_loss+task_loss, latent_representation, decoded, mean, log_variation)

    # def train_model(self,input_batch, target_value, condition, _latent_representation):
    #     self.train()
    #     zeros_direction = torch.zeros(input_batch.size(0), 1, dtype=input_batch.dtype, device=input_batch.device)
    #
    #     reconstruction_loss, decoded, latent_representation, mean, log_variation = self.autoencoder.train_model(input_batch, condition)
    #
    #     if _latent_representation.shape[0] == 0:
    #         _latent_representation = latent_representation
    #     else:
    #         _latent_representation = torch.cat([_latent_representation, latent_representation[-1]])
    #
    #     task_loss, task_pred = self.task_network.train_model(_latent_representation.clone().detach(), target_value, condition)
    #     # task_loss, task_pred = self.task_network.train_model(mean.clone().detach(), target_value, condition.clone().detach())
    #
    #     return (reconstruction_loss, task_loss, reconstruction_loss+task_loss, _latent_representation, decoded, mean, log_variation)
    #
    # def train_model(self, input_batch, target_value, condition):
    #     self.train()
    #
    #     # Forward pass through the autoencoder
    #     decoded, task_pred, encoded, mean, log_variation = self.forward(input_batch, condition, self.reparameterize)
    #
    #     # Calculate the reconstruction loss
    #     reconstruction_loss = self.loss_function(input_batch, decoded, mean, log_variation)
    #
    #     # Calculate the task-specific loss
    #     task_loss = self.criterion(task_pred, target_value)
    #
    #     # Assume you have separate optimizers for encoder, decoder, and task_network
    #     encoder_optimizer.zero_grad()
    #     decoder_optimizer.zero_grad()
    #     task_network_optimizer.zero_grad()
    #
    #     # Backpropagation for the decoder (reconstruction loss)
    #     reconstruction_loss.backward(retain_graph=True)  # retain_graph is True to allow further backprop on shared graphs
    #     decoder_optimizer.step()
    #
    #     # Backpropagation for the task network (task loss)
    #     task_loss.backward()
    #     task_network_optimizer.step()
    #
    #     # Optionally, you might want to update the encoder based on a combination or individually based on each loss
    #     encoder_optimizer.zero_grad()
    #     # Compute some encoder-specific loss or use existing ones
    #     # For demonstration, let's combine both losses for the encoder
    #     combined_loss = reconstruction_loss + task_loss
    #     combined_loss.backward()
    #     encoder_optimizer.step()
    #
    #     self.last_loss = combined_loss.item()
    #     return (reconstruction_loss.item(), task_loss.item(), self.last_loss, encoded, decoded)

    def evaluate(self, input_batch, target_value, condition):
        self.eval()
        self.autoencoder.eval()
        self.task_network.eval()

        with torch.no_grad():
            reconstruction_loss, decoded, latent_representation, mean, log_variation = self.autoencoder.evaluate(input_batch, self.direction)
            # task_loss, task_pred = self.task_network.evaluate(latent_representation, condition)
            task_loss, task_pred = self.task_network.evaluate(mean, condition)

            combined_loss = self.reconstruction_weight*reconstruction_loss + self.task_weight*task_loss
        return (decoded, task_pred), combined_loss, reconstruction_loss, task_loss

    def evaluate_gradient(self, input_batch, target_value, condition):
        self.eval()
        self.autoencoder.eval()
        self.task_network.eval()

        reconstruction_loss, decoded, latent_representation, mean, log_variation = self.autoencoder.evaluate(input_batch, self.direction)
        # task_loss, task_pred = self.task_network.evaluate(latent_representation, condition)
        task_loss, task_pred = self.task_network.evaluate(mean, condition)
        combined_loss = self.reconstruction_weight*reconstruction_loss + self.task_weight*task_loss
        return (decoded, task_pred), combined_loss, reconstruction_loss, task_loss

    # def evaluate_gradient(self, input_batch, target_value, condition):
    #     """
    #     Evaluate the model to compute the loss for the given input batch and target value.
        
    #     Args:
    #     - input_batch (torch.Tensor): The input batch.
    #     - target_value (torch.Tensor): The target value for the task network.
        
    #     Returns:
    #     - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the decoded output and the task prediction.
    #     - torch.Tensor: The combined loss (reconstruction loss + task loss).
    #     """
        
    #     self.eval()
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     self.task_network.eval()
        
    #     decoded, task_pred, encoded, mean, log_variation = self.forward(input_batch, condition, self.reparameterize_evaluate)
    
    #     # Calculate the reconstruction loss
    #     reconstruction_loss = self.loss_function(input_batch, decoded, mean, log_variation)
    
    #     # Forward pass through the task network
    #     task_pred = self.task_network(encoded)
        
    #     # task_loss = self.criterion(task_pred, target_value)
    #     if task_pred < 0:

    #         task_loss = -task_pred**2
    #     else:
    #         task_loss = task_pred**2
    #         # task_loss = 0

    #     # task_loss = (torch.sum(input_batch, dim=1) - target_value) ** 2
    #     # Combine the losses
    #     combined_loss = self.reconstruction_weight*reconstruction_loss + self.task_weight*task_loss

    #     # Return both decoded output and task prediction, and the combined loss
    #     return (decoded, task_pred), combined_loss, reconstruction_loss, task_loss

    @property
    def objective_function_value(self):
        return self.last_loss
    

class TaskNetwork(nn.Module):
    def __init__(self, latent_size, task_layer_sizes, activation=nn.ReLU):
        super(TaskNetwork, self).__init__()
        self.model = nn.Sequential()
        # self.fc = nn.Linear(latent_size, 1)

        for i in range(len(task_layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(task_layer_sizes[i], task_layer_sizes[i+1], bias=True))
            if i < len(task_layer_sizes) - 2:  # Activation after each layer except the last
                self.model.add_module(f"activation_{i}", activation())

        # self.model.add_module(f"activation_{i+1}", nn.Tanh())

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, maximize=False)

    def forward(self, x, condition):
        return self.model(torch.cat((x, condition), dim=-1))

    def train_model(self, latent_space, target_value, condition):
        self.train()
        self.optimizer.zero_grad()
        task_pred = self.forward(latent_space, condition)
        loss = self.criterion(task_pred, target_value)
        loss.backward()
        self.optimizer.step()
        return loss.item(), task_pred

    def evaluate(self, latent_space, condition, target_value=torch.tensor([0], device="cuda:0")):
        self.eval()            
        task_pred = self.forward(latent_space, condition)
        task_loss = self.criterion(torch.tensor([0], device="cuda:0"),task_pred)
        # task_loss = self.criterion(task_pred, target_value)

        task_loss = -1*task_pred.item()
        # print(task_loss)
        return task_loss, task_pred


if __name__ == "__main__":

    config_path = f'./config/AE.yaml'

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict["AntEnv_v3"]

    joints = config["joints"]

    model_kwargs = {'input_size': config['input_size']*joints, 
                    'latent_size': config['latent_size'],
                    'encoder_layer_sizes': config['encoder_layer_sizes'],
                    'decoder_layer_sizes': config['decoder_layer_sizes'],
                    'task_layer_sizes': config["task_layer_sizes"],
                    'condition_size': config["condition_size"]}



    model = AugmentedConditionalVariationalAutoencoder(**model_kwargs)
    print(model)
