
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

class LSTM_Unc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_num):
        super(LSTM_Unc, self).__init__()
        self.hidden_size = 128
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Define the RNN layer
        self.rnn = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        
        # Define a fully connected layer to output the estimated properties
        # self.fc = nn.Sequential(nn.Linear(self.hidden_size, output_size*2))
        linear_sizes = [
            [],
            [64],
            [128, 64],
            [128, 64, 64],
            [128, 128, 64, 64]
        ]
        layers = []
        in_features = self.hidden_size
        for i in range(model_num):
            layers.append(nn.Linear(in_features, linear_sizes[model_num][i]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            in_features = linear_sizes[model_num][i]
        layers.append(nn.Linear(in_features, output_size*2))
        self.fc = nn.Sequential(*layers)
        

        # Sigmoid activation for output normalization to [0, 1]
        # self.output_activation = nn.Sigmoid()
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the RNN
        out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Use the last time step's output for property estimation
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        return self.output_activation(out[:,:self.output_size]), out[:,self.output_size:]