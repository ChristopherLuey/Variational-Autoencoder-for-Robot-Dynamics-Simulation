import torch
from ae import AugmentedAutoencoder
from ae import TaskNetwork
import matplotlib.pyplot as plt


class AutoencoderPipeline:

    def __init__(self):
        self.control_sequence_size = 32
        self.latent_size = 3
        self.encoder_layer_sizes = [15]
        self.decoder_layer_sizes = [15]
        self.autoencoder = AugmentedAutoencoder(self.control_sequence_size, self.latent_size, self.encoder_layer_sizes, self.decoder_layer_sizes)
        self.task_network = TaskNetwork(self.latent_size)



        self.target_value_tensor = torch.tensor([0], dtype=torch.float32, device="cpu")
        self.N = 100
        self.perturbation_strength = 0.05
        print("\n\n\n")
        print("Target Value = {}".format(self.target_value_tensor))
        print("Samples = {}".format(self.N))
        print("Perturbation Size = {}".format(self.perturbation_strength))
        print("\n\n\n")


    def acquire_new_data(self, last_control_seq):
        """
        Acquire new data for training the autoencoder.

        Returns:
        - The new control sequence with the lowest autoencoder objective function value after sampling.
        """
        lowest_loss = float('inf')
        best_seq = None

        for _ in range(self.N):
            # Perturb the control sequence by a random vector
            perturbation = (torch.rand_like(last_control_seq)-0.5) * self.perturbation_strength
            perturbed_seq = torch.clamp(last_control_seq + perturbation, min=0, max=1)

            # Evaluate the perturbed sequence without updating the autoencoder weights
            _, loss = self.autoencoder.evaluate(perturbed_seq.unsqueeze(0), self.target_value_tensor)

            if loss < lowest_loss:
                lowest_loss = loss
                best_seq = perturbed_seq

        return best_seq
    
    
    def train_model_pipeline(self, epochs):
        losses = []
        obj_loss = []
        new_control_seq_values = []
        new_control_seq = torch.rand(self.control_sequence_size)
        print(new_control_seq)
        for i in range(epochs):
            new_control_seq = self.acquire_new_data(new_control_seq)
            loss = self.autoencoder.train_model(new_control_seq.unsqueeze(0), self.target_value_tensor)
            losses.append(loss[2])
            obj_loss.append(loss[1])
            new_control_seq_values.append(new_control_seq.tolist()) 
        
        print(new_control_seq)
        print(losses[-1])
        print(obj_loss[-1])

        # # Plotting losses
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1)  # First subplot for the losses
        # plt.plot(range(1, epochs + 1), losses, label="Training Loss")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Model Loss Over Time')
        # plt.legend()

        # plt.subplot(1, 3, 2)  # First subplot for the losses
        # plt.plot(range(1, epochs + 1), obj_loss, label="Training Loss")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Objective Loss Over Time')
        # plt.legend()

        # # Plotting values of new_control_seq over time
        # plt.subplot(1, 3, 3)  # Second subplot for the new_control_seq values
        # new_control_seq_values = list(zip(*new_control_seq_values))
        # for seq_index, seq_values in enumerate(new_control_seq_values):
        #     plt.plot(range(1, epochs + 1), seq_values, label=f"Seq {seq_index + 1}")
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.title('New Control Seq Values Over Time')
        # plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Plot training loss on the first subplot
        axes[0].plot(range(1, epochs + 1), losses, label="Training Loss")
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Time')
        axes[0].legend()

        # Plot objective loss on the second subplot
        axes[1].plot(range(1, epochs + 1), obj_loss, label="Objective Loss")
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Objective Loss Over Time')
        axes[1].legend()

        # Plot new control sequence values on the third subplot
        new_control_seq_values_transposed = list(zip(*new_control_seq_values))
        for seq_index, seq_values in enumerate(new_control_seq_values_transposed):
            axes[2].plot(range(1, epochs + 1), seq_values, label=f"Seq {seq_index + 1}")
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Value')
        axes[2].set_title('New Control Seq Values Over Time')
        #axes[2].legend()

        plt.tight_layout()
        plt.show()

# Binary cross entropy loss for binary inputs
# Sigmoid Acctivation in decoder output

if __name__ == "__main__":
    aep = AutoencoderPipeline()
    epochs = 1000

    aep.train_model_pipeline(epochs)
