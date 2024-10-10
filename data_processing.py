import torch
import matplotlib.pyplot as plt


def split_data(controls, conditions, train_ratio=0.8, device='cpu'):
    controls_tensor = torch.stack(controls).to(device)
    conditions_tensor = torch.stack(conditions).to(device)
    train_size = int(train_ratio * len(controls_tensor))
    val_size = len(controls_tensor) - train_size
    train_controls, val_controls = torch.utils.data.random_split(controls_tensor, [train_size, val_size])
    train_conditions, val_conditions = torch.utils.data.random_split(conditions_tensor, [train_size, val_size])
    return train_controls, val_controls, train_conditions, val_conditions


def train_autoencoder(autoencoder, train_controls, train_conditions, val_controls, val_conditions, device, epochs=100, visualize_reconstruction=False):
    weights = torch.ones(train_controls[0].shape).to(device)
    train_losses = []
    val_losses = []
    latent_space_data = []

    for epoch in range(epochs):
        loss, decoded, latent_representation, mean, log_variation = autoencoder.train_model(train_controls, train_conditions, weights, device)
        train_losses.append(loss)
        latent_space_data.append(latent_representation.detach().cpu().numpy())
        print(f"Epoch {epoch + 1}, Training Loss: {loss}")

        # Validation step
        val_loss, _, _, _, _ = autoencoder.evaluate(val_controls, val_conditions, device)
        val_losses.append(val_loss.item())
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss.item()}")
        
        if visualize_reconstruction:
            visualize_validation(autoencoder, val_controls, val_conditions, device)

    autoencoder.plot_latent_space_dynamics(latent_space_data)
    return train_losses, val_losses


def plot_training_validation_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()


def plot_latent_space_characteristics(train_conditions):
    latent_means = [mean.mean().item() for mean in train_conditions]
    latent_vars = [torch.exp(0.5 * log_variation).mean().item() for log_variation in train_conditions]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(latent_means) + 1), latent_means, label='Latent Mean')
    plt.plot(range(1, len(latent_vars) + 1), latent_vars, label='Latent Variance')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Latent Space Characteristics Over Training')
    plt.legend()
    plt.show()


def visualize_validation(autoencoder, val_controls, val_conditions, device):
    autoencoder.eval()
    with torch.no_grad():
        for i in range(min(5, len(val_controls))):  # Visualize up to 5 examples
            val_control = val_controls[i].to(device)
            val_condition = val_conditions[i].to(device)
            _, decoded, _, _, _ = autoencoder.forward(val_control, val_condition, evaluate=True)
            
            plt.figure(figsize=(10, 4))
            plt.plot(val_control.cpu().numpy(), label='Original Control Sequence')
            plt.plot(decoded.cpu().numpy(), label='Reconstructed Control Sequence')
            plt.title(f'Validation Sample {i + 1}')
            plt.xlabel('Time Step')
            plt.ylabel('Control Value')
            plt.legend()
            plt.show()
