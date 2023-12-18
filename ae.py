import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

class AE_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)



class Encoder(nn.Module):
    def __init__(self, input_dimension, network_slope, latent_space_dimension, activation_function=nn.GELU):
        super().__init__()




class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        

