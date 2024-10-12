import torch
import matplotlib.pyplot as plt

def split_data(controls, conditions, train_ratio=0.8, device='cpu'):
    if not isinstance(controls, torch.Tensor):
        raise TypeError(f"Unsupported type for controls: {type(controls)}")
    if not isinstance(conditions, torch.Tensor):
        raise TypeError(f"Unsupported type for conditions: {type(conditions)}")

    controls_tensor = controls.to(device)
    conditions_tensor = conditions.to(device)

    dataset_size = controls_tensor.size(0)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_controls = controls_tensor[train_indices]
    val_controls = controls_tensor[val_indices]
    train_conditions = conditions_tensor[train_indices]
    val_conditions = conditions_tensor[val_indices]

    return train_controls, val_controls, train_conditions, val_conditions
