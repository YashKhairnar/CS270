import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, numClasses=12):
        super(CNNModel, self).__init__()
        layers = 3
        fclayers = 128
        in_channels = 3
        out_channels = 64
        conv_blocks = []

        # Build convolution layers
        for _ in range(layers):
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(2))

            in_channels = out_channels
            out_channels *= 2   # double filters each layer

        #convolution layer
        self.conv_layers = nn.Sequential(*conv_blocks)
        
        final_image_size = 224 // (2 ** layers)
        flattened_size = in_channels * final_image_size * final_image_size

        #fully connected neural network layer
        self.fc_layer = nn.Sequential(
            nn.Linear(flattened_size, fclayers),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fclayers, numClasses)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # [B, C*H*W]
        x = self.fc_layer(x)
        return x
