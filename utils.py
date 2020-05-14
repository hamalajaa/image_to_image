import torch
import torch.nn as nn

def init_weights(m, scale_factor=0.02):
    if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.normal_(m.weight.data, 0.0, scale_factor)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, scale_factor)
        nn.init.constant_(m.bias.data, 0.0)


def set_requires_grad(nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad