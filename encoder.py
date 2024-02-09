import torch.nn as nn
class Encoder(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
