import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
import json
import yaml
import gym
import matplotlib.pyplot as plt

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
    def __init__(self, input_size, latent_size, encoder_layer_sizes, decoder_layer_sizes, condition_size, output_activation=None, learning_rate=1e-2):
        super(BasicAutoencoder, self).__init__()
        full_encoder_sizes = [input_size + condition_size] + encoder_layer_sizes
        self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)
        full_decoder_sizes = [latent_size + condition_size] + decoder_layer_sizes + [input_size]
        self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x, condition, evaluate=False):
        mean, log_variation = self.encoder(torch.cat((x, condition), dim=-1))
        if evaluate:
            latent_representation = self.reparameterize_evaluate(mean, log_variation)
        else:
            latent_representation = self.reparameterize(mean, log_variation)
        decoded = self.decoder(torch.cat((latent_representation, condition), dim=-1))
        return decoded, latent_representation, mean, log_variation

    def reparameterize(self, mean, log_variation):
        std = torch.exp(0.5 * log_variation) # convert log to non-log
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize_evaluate(self, mean, log_variation):
        return mean

    def loss_function(self, input_batch, decoded, mean, log_variation, normalized_weights):
        reproduction_loss = torch.mean(normalized_weights * self.criterion(decoded, input_batch))
        KLD = -0.5 * torch.sum(1 + log_variation - mean.pow(2) - log_variation.exp())
        return reproduction_loss + KLD

    def train_model(self, input_batch, condition, weights, device):
        self.train()
        input_batch = input_batch.to(device)
        condition = condition.to(device)
        weights = weights.to(device)
        decoded, latent_representation, mean, log_variation = self.forward(input_batch, condition)
        loss = self.loss_function(input_batch, decoded, mean, log_variation, weights)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping added
        self.optimizer.step()
        return loss.item(), decoded, latent_representation, mean, log_variation

    def evaluate(self, input_batch, condition, device):
        self.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device)
            condition = condition.to(device)
            decoded, latent_representation, mean, log_variation = self.forward(input_batch, condition, evaluate=True)
            loss = self.loss_function(input_batch, decoded, mean, log_variation, torch.ones_like(input_batch))
        return loss, decoded, latent_representation, mean, log_variation

    def plot_training_validation_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    def plot_latent_space_dynamics(self, latent_space_data):
        plt.figure(figsize=(10, 5))
        for epoch, latent_data in enumerate(latent_space_data):
            plt.scatter([epoch + 1] * len(latent_data), latent_data, alpha=0.5, label=f'Epoch {epoch + 1}' if epoch < 5 else "")
        plt.xlabel('Epoch')
        plt.ylabel('Latent Space Value')
        plt.title('Latent Space Dynamics Over Training')
        plt.legend()
        plt.show()