import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(ProtoNet, self).__init__()
        self.embedding = nn.Sequential(
            conv_block(in_channels, out_channels),
            conv_block(out_channels, out_channels),
            conv_block(out_channels, out_channels),
            conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(x.size(0), -1)
        return out
