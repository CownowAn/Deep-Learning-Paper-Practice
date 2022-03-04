from typing import OrderedDict
from sympy import Order
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d, MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, **kwargs):
    """
    4 modules with a 3x3 convolutions and hidden_size(default=64) filters, 
    followed by batch normalization, 
    a ReLU nonlinearity, 
    and 2x2 max-pooling

    Args:
        in_channels (int): number of channels before the MetaConv2d layer
        out_channels (int): number of channels after the MetaConv2d layer

    Returns:
        MetaSequential (OrderedDict): 4 modules with conv, norm, relu, pool layers.
    """
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaConvModel(MetaModule):
    """
    Args:
        in_channels (int): the number of channels for the input images.
        out_features (int): the number of classes.
        hidden_size (int): the number of channels in the intermediate representations. (default=64)
        feature_size (int): the number of features returned by the convolutional head. (default=64)
    """
    
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        
        # MetaSequential is similar to nn.Sequential
        # A sequential container.
        # Modules will be added to it in the order they are passed in the constructor.
        # like in here MetaConv2D is passed as convulational layer and then a ReLU, MaxPool.
        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        
        # MetaLinear is similar to torch.nn.Linear
        # Applies a linear transformation to the incoming data
        self.classifier = MetaLinear(feature_size, out_features, bias=True)
    
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
    
def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size, feature_size=hidden_size)
    
def ModelConvMiniImagenet(out_features, hidden_size=32):
    return MetaConvModel(3, out_features, hidden_size=hidden_size, feature_size=5*5*hidden_size)

if __name__ == '__main__':
    model = ModelConvOmniglot()