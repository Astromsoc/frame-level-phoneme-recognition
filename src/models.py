"""
    MLP for frame-level phoeneme classification
"""

import torch
from torch import nn
from itertools import zip_longest


class MLP(torch.nn.Module):

    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(),
        'softplus': nn.Softplus()
    }
    output_dim = 40

    def __init__(
        self, 
        dim_list: list, 
        activation_list: list,
        dropout_list: list,
        batchnorm_list: list,
        add_squares: bool = True
    ):
        super(MLP, self).__init__()

        # fundamental prerequisite checks
        assert len(dim_list) - 1 == len(activation_list) == len(dropout_list)

        # whether to kernelize inputs with squares
        self.add_squares = add_squares
        # double the input dim if squares are added
        if self.add_squares:
            dim_list[0] *= 2

        # add the unchanged last layer
        dim_list.append(self.output_dim)

        self.linears = [
            nn.Linear(dim_list[i], dim_list[i + 1])
            for i in range(len(dim_list) - 1)
        ]
        self.activations = [
            self.activation_map[a] for a in activation_list
        ]
        self.dropouts = [
            nn.Dropout(dropout_rate) if dropout_rate else None 
            for dropout_rate in dropout_list 
        ]
        self.batchnorms = ([
            nn.BatchNorm1d(dim) if batchnorm_list[i] else None
            for i, dim in enumerate(dim_list[1:-1]) 
        ])
        
        # build the layers given input configurations
        layers = [
            unit for pair in zip_longest(self.linears, self.batchnorms, self.activations, self.dropouts) 
            for unit in pair if unit 
        ]
        # alter the order of dropout and relu if they are intertwined
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Dropout):
                if i > 0 and isinstance(layers[i - 1], nn.ReLU):
                    layers[i - 1], layers[i] = layers[i], layers[i - 1]
        # build the model accordingly
        self.streamline = nn.Sequential(*layers)

        # compute the parameter size
        self.trainable_param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_param_count = sum(p.numel() for p in self.parameters())
        

    def forward(self, x):
        # flatten the input
        x = x.view(x.size(0), -1)
        if self.add_squares:
            x = torch.hstack((x, torch.square(x)))
        return self.streamline(x)
