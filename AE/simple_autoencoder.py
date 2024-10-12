import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    def __init__(self, joints, timesteps, latent_size, layer_sizes, condition_size, output_activation=None, learning_rate=1e-3):
        super(BasicAutoencoder, self).__init__()
        self.joints = joints
        self.timesteps = timesteps
        self.condition_size = condition_size
        input_size = joints * timesteps
        full_encoder_sizes = [input_size + condition_size] + layer_sizes
        self.encoder = CVAEEncoder(full_encoder_sizes, latent_size)
        full_decoder_sizes = [latent_size + condition_size] + layer_sizes[::-1] + [input_size]
        self.decoder = CVAEDecoder(full_decoder_sizes, latent_size, output_activation=output_activation)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')  # Changed to 'mean' for stability

    def save_model_architecture(self, file_path):
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, 'model_architecture.txt')
        with open(file_path, 'w') as f:
            f.write(str(self))
        print(f"Model architecture saved to {file_path}")

    @classmethod
    def load_model_architecture(cls, file_path, learning_rate=1e-3):
        with open(file_path, 'r') as f:
            model_str = f.read()
        # Extract the parameters from the model string
        # This is a placeholder, actual implementation will depend on how the model string is formatted
        joints = ...  # Extract joints from model_str
        timesteps = ...  # Extract timesteps from model_str
        latent_size = ...  # Extract latent_size from model_str
        layer_sizes = ...  # Extract layer_sizes from model_str
        condition_size = ...  # Extract condition_size from model_str
        output_activation = ...  # Extract output_activation from model_str if available
        return cls(joints, timesteps, latent_size, layer_sizes, condition_size, output_activation, learning_rate)

    def forward(self, x, condition=None, evaluate=False):
        if x.dim() != 2:
            raise ValueError(f"Expected input tensor to have 2 dimensions [batch_size, input_size], but got {x.dim()} dimensions.")

        input_size = x.size(1)
        expected_input_size = self.encoder.model[0].in_features - self.condition_size
        if input_size != expected_input_size:
            raise ValueError(f"Input size ({input_size}) does not match expected size ({expected_input_size}).")

        if condition is not None:
            condition = condition.to(x.device)
            x = torch.cat((x, condition), dim=-1)

        mean, log_variation = self.encoder(x)
        latent_representation = mean if evaluate else self.reparameterize(mean, log_variation)

        if condition is not None:
            latent_representation = torch.cat((latent_representation, condition), dim=-1)
        
        decoded = self.decoder(latent_representation)
        return decoded, latent_representation, mean, log_variation

    def reparameterize(self, mean, log_variation):
        std = torch.exp(0.5 * log_variation)
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize_evaluate(self, mean, log_variation):
        return mean

    def loss_function(self, input_batch, decoded, mean, log_variation):
        reproduction_loss = self.criterion(decoded, input_batch)
        KLD = -0.5 * torch.mean(1 + log_variation - mean.pow(2) - log_variation.exp())
        return reproduction_loss + KLD

    def train_single_epoch(self, train_controls, condition, device):
        self.train()
        train_controls = train_controls.to(device)  # Shape: [batch_size, input_size]
        condition = condition.to(device) if condition is not None else None
        self.optimizer.zero_grad()
        
        # Debug: Print shapes and check for NaNs
        print(f"Train controls shape: {train_controls.shape}")
        print(f"Condition shape: {condition.shape if condition is not None else None}")
        print(f"Any NaN in train_controls: {torch.isnan(train_controls).any()}")
        
        decoded, latent_representation, mean, log_variation = self.forward(train_controls, condition)
        
        # Debug: Print shapes and check for NaNs after forward pass
        print(f"Decoded shape: {decoded.shape}")
        print(f"Mean shape: {mean.shape}")
        print(f"Log variation shape: {log_variation.shape}")
        print(f"Any NaN in decoded: {torch.isnan(decoded).any()}")
        print(f"Any NaN in mean: {torch.isnan(mean).any()}")
        print(f"Any NaN in log_variation: {torch.isnan(log_variation).any()}")
        
        loss = self.loss_function(train_controls, decoded, mean, log_variation)
        
        # Debug: Print loss and check for NaN
        print(f"Loss: {loss.item()}")
        print(f"Is loss NaN: {torch.isnan(loss).any()}")
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return loss.item(), latent_representation

    def train_model(self, train_controls, val_controls, device, epochs=100, visualize_reconstruction=False, results_dir=None, condition=None):
        train_losses = []
        val_losses = []
        latent_space_data = []

        # Verify the shape of train_controls and val_controls
        if train_controls.dim() != 3 or val_controls.dim() != 3:
            raise ValueError("train_controls and val_controls must have 3 dimensions [batch_size, timesteps, joints]")
        if train_controls.size(1) != self.timesteps or train_controls.size(2) != self.joints:
            raise ValueError(f"train_controls must have shape [batch_size, {self.timesteps}, {self.joints}]")
        if val_controls.size(1) != self.timesteps or val_controls.size(2) != self.joints:
            raise ValueError(f"val_controls must have shape [batch_size, {self.timesteps}, {self.joints}]")

        # Convert to [batch_size, timesteps * joints]
        train_controls = train_controls.view(train_controls.size(0), -1)
        val_controls = val_controls.view(val_controls.size(0), -1)

        # Save the model architecture to the results directory at the start
        if results_dir is not None:
            self.save_model_architecture(results_dir)

        for epoch in range(epochs):
            if train_controls.size(0) == 0:
                raise ValueError("Training data is empty. Please provide valid training data.")
            train_loss, latent_representation = self.train_single_epoch(train_controls, condition, device)
            train_losses.append(train_loss)
            latent_space_data.append(latent_representation.detach().cpu().numpy())
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

            # Validation step
            val_loss, _, _, _, _ = self.evaluate(val_controls, device, condition)
            val_losses.append(val_loss.item())
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss.item()}")

            if visualize_reconstruction and results_dir is not None:
                epoch_dir = os.path.join(results_dir, f'epoch_{epoch + 1}')
                os.makedirs(epoch_dir, exist_ok=True)
                self.visualize_validation(val_controls, device, epoch_dir, condition)

        self.plot_latent_space_dynamics(latent_space_data, results_dir)
        self.plot_training_validation_losses(train_losses, val_losses, results_dir)

        # Save the trained model to the results directory
        if results_dir is not None:
            torch.save(self.state_dict(), os.path.join(results_dir, 'trained_autoencoder.pth'))

        return train_losses, val_losses

    def evaluate(self, input_batch, device, condition=None):
        self.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device)  # Shape: [batch_size, input_size]
            condition = condition.to(device) if condition is not None else None
            decoded, latent_representation, mean, log_variation = self.forward(input_batch, condition, evaluate=True)
            loss = self.loss_function(input_batch, decoded, mean, log_variation)
        return loss, decoded, latent_representation, mean, log_variation

    def plot_training_validation_losses(self, train_losses, val_losses, results_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'training_validation_losses.png'))
        plt.close()

    def plot_latent_space_dynamics(self, latent_space_data, results_dir):
        plt.figure(figsize=(10, 5))
        for epoch, latent_data in enumerate(latent_space_data):
            plt.scatter([epoch + 1] * len(latent_data), latent_data, alpha=0.5, label=f'Epoch {epoch + 1}' if epoch < 5 else "")
        plt.xlabel('Epoch')
        plt.ylabel('Latent Space Value')
        plt.title('Latent Space Dynamics Over Training')
        if latent_space_data and len(latent_space_data) > 0:
            plt.legend()
        plt.savefig(os.path.join(results_dir, 'latent_space_dynamics.png'))
        plt.close()

    def visualize_validation(self, val_controls, device, results_dir, condition=None):
        self.eval()
        with torch.no_grad():
            val_controls = val_controls.to(device)
            condition = condition.to(device) if condition is not None else None
            decoded, _, _, _ = self.forward(val_controls, condition)
            for i in range(min(5, decoded.size(0))):  # Visualize up to 5 examples
                original = val_controls[i].cpu().numpy()
                reconstruction = decoded[i].cpu().numpy()

                plt.figure(figsize=(10, 4))
                # Assuming input_size = timesteps * joints
                time_steps = self.timesteps
                if time_steps * self.joints != original.shape[0]:
                    raise ValueError(f"Input size is not divisible by number of joints ({self.joints}).")
                original = original.reshape(time_steps, self.joints)
                reconstruction = reconstruction.reshape(time_steps, self.joints)

                for joint in range(original.shape[1]):
                    plt.plot(original[:, joint], label=f'Original Joint {joint+1}' if joint == 0 else "")
                    plt.plot(reconstruction[:, joint], linestyle='--', label=f'Reconstructed Joint {joint+1}' if joint == 0 else "")
                plt.title(f'Validation Sample {i + 1}')
                plt.xlabel('Time Step')
                plt.ylabel('Control Value')
                plt.legend()
                plt.savefig(os.path.join(results_dir, f'validation_sample_{i + 1}.png'))
                plt.close()

    def load_weights(self, weights_path):
        """
        Load weights from a file.
        
        Args:
            weights_path (str): Path to the weights file.
        """
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"No such file: '{weights_path}'")
        
        self.load_state_dict(torch.load(weights_path))
        print(f"Weights loaded from '{weights_path}'")
