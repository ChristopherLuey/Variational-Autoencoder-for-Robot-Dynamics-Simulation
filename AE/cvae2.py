import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

class ContrastiveCVAEEncoder(nn.Module):
    def __init__(self, layer_sizes, latent_space, activation=nn.ReLU):
        super(ContrastiveCVAEEncoder, self).__init__()
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


class ContrastiveCVAEDecoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, activation=nn.ReLU, output_activation=None):
        super(ContrastiveCVAEDecoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: 
                self.model.add_module(f"activation_{i}", activation())
        if output_activation is not None:
            self.model.add_module("output_activation", output_activation)

    def forward(self, x):
        return self.model(x)


class ContrastiveBasicAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, condition_size, output_activation=None):
        super(ContrastiveBasicAutoencoder, self).__init__()
        full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        self.encoder = ContrastiveCVAEEncoder(full_encoder_sizes, latent_size)
        full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        self.decoder = ContrastiveCVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)
        self.learning_rate = 1e-2
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

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

    def loss_function(self, input_batch, decoded, mean, log_variation, normalized_weights):
        reproduction_loss = torch.mean(normalized_weights * self.criterion(decoded, input_batch))
        #reproduction_loss = torch.mean(normalized_weights * (input_batch - decoded).pow(2))
        #reproduction_loss = torch.mean(normalized_weights * reproduction_loss)

        KLD = -0.0 * torch.sum(1+ log_variation - mean.pow(2) - log_variation.exp())
        return reproduction_loss + KLD

    def train_model(self, input_batch1, input_batch2, condition1, condition2, weights):
        self.train()
        decoded1, latent_representation1, mean1, log_variation1= self.forward(input_batch1, condition1)
        decoded2, latent_representation2, mean2, log_variation2= self.forward(input_batch2, condition2)

        reconstruction1 = self.criterion(decoded1,input_batch1)
        reconstruction2 = self.criterion(decoded2,input_batch2)

        margin = 1.0

        euclidean_distance = F.pairwise_distance(latent_representation1, latent_representation2)
        contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        #loss = self.loss_function(input_batch, decoded, mean, log_variation, weights)
        loss = reconstruction1 + reconstruction2 + contrastive
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), decoded, latent_representation, mean, log_variation

    def evaluate(self,input_batch, condition):
        self.eval()
        decoded, latent_representation, mean, log_variation = self.forward(input_batch, condition)
        loss = self.loss_function(input_batch, decoded, mean, log_variation, 1)
        return loss, decoded, latent_representation, mean, log_variation


class ContrastiveAugmentedConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, task_layer_sizes, condition_size, output_activation=None):
        super(ContrastiveAugmentedConditionalVariationalAutoencoder, self).__init__()
        # Define encoder layer sizes including the input and latent layers
        # full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        # self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)

        # # Define decoder layer sizes
        # full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        # self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)

        # Define task network

        self.autoencoder = ContrastiveBasicAutoencoder(input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, condition_size)

        task_layer_sizes[0] = task_layer_sizes[0] + condition_size
        self.task_network = ContrastiveTaskNetwork(latent_size, task_layer_sizes)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss for non-binary data
        self.last_loss = None
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer_autoencoder = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        self.optimizer_task_network = optim.Adam(self.task_network.parameters(), lr=1e-2)
        self.reconstruction_weight = [0.8, 1]
        self.task_weight = [0.2,0]

        self.direction = torch.tensor([0.0], dtype=torch.float32, device="cuda:0")
        self.training_epochs = 0

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

    # def train_model(self, input_batch, target_value, condition, _latent_representation):
    #     # for layer in self.children():
    #     #     if hasattr(layer, 'reset_parameters'):
    #     #         layer.reset_parameters()
    #     # for layer in self.task_network.children():
    #     #     if hasattr(layer, 'reset_parameters'):
    #     #         layer.reset_parameters()
    #     #
    #     # for layer in self.autoencoder.encoder.children():
    #     #     if hasattr(layer, 'reset_parameters'):
    #     #         layer.reset_parameters()
    #     # for layer in self.autoencoder.decoder.children():
    #     #     if hasattr(layer, 'reset_parameters'):
    #     #         layer.reset_parameters()
    #     #
    #     # print(self.task_network.parameters())
    #
    #
    #     self.train()
    #     self.autoencoder.encoder.train()
    #     self.autoencoder.decoder.train()
    #     self.task_network.train()
    #     # Zero the gradientss
    #     self.optimizer.zero_grad()
    #
    #     # Forward pass through the autoencoder
    #     decoded, latent_representation, mean, log_variation = self.autoencoder.forward(input_batch, condition)
    #
    #     # Calculate the reconstruction loss
    #     reconstruction_loss = self.autoencoder.loss_function(input_batch, decoded, mean, log_variation)
    #     # if _latent_representation.shape[0] == 0:
    #     #     _latent_representation = latent_representation
    #     # else:
    #     #     _latent_representation = torch.cat([_latent_representation, latent_representation[-1]])
    #     # Calculate the task-specific loss
    #     # task_pred = self.task_network.forward(_latent_representation, condition)
    #
    #     if _latent_representation.shape[0] == 0:
    #         _latent_representation = mean
    #     else:
    #         _latent_representation = torch.cat([_latent_representation, mean[-1]])
    #
    #     task_pred = self.task_network.forward(_latent_representation, condition)
    #     # task_pred = self.task_network.forward(mean, condition)
    #
    #     task_loss = self.task_network.criterion(task_pred, target_value)
    #
    #     # Combine the losses
    #     combined_loss = reconstruction_loss * self.reconstruction_weight[0] + task_loss * self.task_weight[0]
    #
    #     # Backward pass and optimize
    #     combined_loss.backward()
    #     self.optimizer.step()
    #
    #     self.last_loss = combined_loss.item()
    #     return (reconstruction_loss, task_loss, reconstruction_loss+task_loss, latent_representation, decoded, mean, log_variation)

    def train_model(self,input_batch1, input_batch2, target_value1, target_value2, condition1, condition2, _latent_representation, iter=1):
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

        # # Find indices where target_value is greater than 0
        # positive_indices = target_value > 0
        # # Find indices where target_value is less than 0
        # negative_indices = target_value < 0
        #
        # # Split input_batch and condition based on positive_indices
        # input_batch_positive = input_batch[positive_indices]
        # condition_positive = condition[positive_indices]
        #
        # # Split input_batch and condition based on negative_indices
        # input_batch_negative = input_batch[negative_indices]
        # condition_negative = condition[negative_indices]

        weights_clamped = torch.clamp(target_value, min=0, max=1)
        # weights_clamped = target_value
        max_weight = weights_clamped.max()
        if max_weight > 0:
            normalized_weights = weights_clamped / max_weight
        else:
            normalized_weights = weights_clamped

        # if target_value < 0:
        #     normalized_weights = -torch.divide(target_value, target_value)
        # else:
        #     normalized_weights = torch.divide(target_value, target_value)

        for i in range(iter):

            self.train()
            #zeros_direction = torch.zeros(input_batch.size(0), 1, dtype=input_batch.dtype, device=input_batch.device)

            reconstruction_loss, decoded, latent_representation, mean, log_variation = self.autoencoder.train_model(input_batch1, input_batch2, condition1, condition2)
            #
            if _latent_representation.shape[0] == 0:
                _latent_representation = latent_representation
            else:
                _latent_representation = torch.cat([_latent_representation, latent_representation[-1]])

            # task_loss, task_pred = self.task_network.train_model(_latent_representation.clone().detach(), target_value, condition)
            # task_loss = 0
            task_loss, task_pred = self.task_network.train_model(mean.clone().detach(), target_value, condition.clone().detach(), normalized_weights)

        self.training_epochs+=1

        return (reconstruction_loss, task_loss, reconstruction_loss+task_loss, _latent_representation, decoded, mean, log_variation)

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

            combined_loss = self.reconstruction_weight[1]*reconstruction_loss + self.task_weight[1]*task_loss
        return (decoded, task_pred), combined_loss, reconstruction_loss, task_loss, mean, log_variation

    def evaluate_gradient(self, input_batch, target_value, condition):
        self.eval()
        self.autoencoder.eval()
        self.task_network.eval()

        reconstruction_loss, decoded, latent_representation, mean, log_variation = self.autoencoder.evaluate(input_batch, self.direction)
        # task_loss, task_pred = self.task_network.evaluate(latent_representation, condition)
        task_loss, task_pred = self.task_network.evaluate(mean, condition, target_value)
        combined_loss = self.reconstruction_weight[1]*reconstruction_loss + self.task_weight[1]*task_loss
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
    

class ContrastiveTaskNetwork(nn.Module):
    def __init__(self, latent_size, task_layer_sizes, activation=nn.LeakyReLU):
        super(ContrastiveTaskNetwork, self).__init__()
        self.model = nn.Sequential()
        # self.fc = nn.Linear(latent_size, 1)

        for i in range(len(task_layer_sizes) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(task_layer_sizes[i], task_layer_sizes[i+1], bias=True))
            if i < len(task_layer_sizes) - 2:  # Activation after each layer except the last
                self.model.add_module(f"activation_{i}", activation())

        # self.model.add_module(f"activation_{i+1}", nn.Sigmoid())

        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x, condition):
        return self.model(torch.cat((x, condition), dim=-1))

    def train_model(self, latent_space, target_value, condition, normalized_weights):
        self.train()
        self.optimizer.zero_grad()
        task_pred = self.forward(latent_space, condition)

        loss = torch.mean(normalized_weights*self.criterion(task_pred, target_value))

        loss.backward()
        self.optimizer.step()
        return loss.item(), task_pred

    def evaluate(self, latent_space, condition, target_value=torch.tensor([0], device="cuda:0")):
        self.eval()            
        task_pred = self.forward(latent_space, condition)
        # task_loss = self.criterion(torch.tensor([0], device="cuda:0"),task_pred)
        # task_loss = torch.mean(self.criterion(task_pred, target_value))

        task_loss = -1*task_pred
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



    model = ContrastiveAugmentedConditionalVariationalAutoencoder(**model_kwargs)
    print(model)
