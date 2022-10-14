import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary


# pytorch normalization select function
def normalize_function(norm_char, num_input_features):
    if norm_char == 'BN':
        norm_func = nn.BatchNorm3d(num_input_features)
    elif norm_char == 'IN':
        norm_func = nn.InstanceNorm3d(num_input_features)

    return norm_func


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, norm_char):
        super().__init__()
        inner_channels = 2 * out_channels

        self.residual = nn.Sequential(
            normalize_function(norm_char, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            normalize_function(norm_char, inner_channels),
            nn.ReLU(),
            nn.Conv3d(inner_channels, out_channels - in_channels, 3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm_char='BN'):
        super().__init__()
        self.add_module('norm1', normalize_function(norm_char, num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', normalize_function(norm_char, bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, norm_char='BN'):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, norm_char)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_char='BN'):
        super().__init__()
        self.add_module('norm', normalize_function(norm_char, num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 norm_char='BN',
                 ):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', normalize_function(norm_char, num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))

        # Each denseblock
        num_features = num_init_features

        self.features = nn.Sequential(OrderedDict(self.features))
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                norm_char=norm_char)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    norm_char=norm_char)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2
        # Final batch norm
        self.features.add_module('norm5', normalize_function(norm_char, num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # self.classifier = nn.Sequential(nn.Conv2d(1020, 1024, (1, 2), stride=(1, 1), padding=(0,0)),
        #                                 nn.ConvTranspose2d(1024, 256, (6, 1), stride=(3, 1), padding=(1, 0))
        #                                 )
        # self.classifier = nn.Sequential(nn.Conv2d(1020, 1024, (5, 3), stride=(4, 2), padding=(2,0)),
        #                                 nn.ConvTranspose2d(1024, 2*256, (6, 1), stride=(3, 1), padding=(1, 0))
        #                                 )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # final_fea = out
        # final_fea = torch.flatten(final_fea, start_dim=2, end_dim=3)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        # coor_val = self.classifier(final_fea)

        return out


def generate_DenseNet(model_depth, n_input_channels, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(n_input_channels=n_input_channels,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(n_input_channels=n_input_channels,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(n_input_channels=n_input_channels,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(n_input_channels=n_input_channels,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    return model


def DenseNet121(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)


def DenseNet169(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)


def DenseNet201(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)


def DenseNet264(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = generate_DenseNet(121, n_input_channels=1, num_classes=3, num_init_features=64,
                              conv1_t_stride=2, norm_char='BN')
    model = model.to(device)

    a = torch.ones(1, 1, 256, 256, 256).to(device)

    b = model(a)

    summary(model,(1,256,256,256),batch_size=2)
