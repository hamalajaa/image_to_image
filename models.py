import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class UnetLayer(nn.Module):
    
    def __init__(self, hps, input_c, inner_f, outer_f, layer_type, sublayer, dropout=False):
        
        """
        Parameters:
            input_c (int)        : number of input channels
            inner_f (int)        : number of filters on inner layer
            outer_f (int)        : number of filters on outer layer
            layer_type (str)     : type of this layer, either 'default', 'innermost' or 'outermost'
            sublayer (UnetLayer) : sublayer in unet-hierarchy
            dropout (bool)       : use dropout layer?
        """
        
        super(UnetLayer, self).__init__()
        
        self.layer_type = layer_type
        
        down_conv = nn.Conv2d(input_c, inner_f, kernel_size=hps.unet_conv_kernel,
                              stride=hps.unet_conv_stride, padding=hps.unet_conv_pad)
        down_relu = nn.LeakyReLU(hps.unet_lrelu_slope, inplace=True)
        down_norm = nn.BatchNorm2d(inner_f)
        
        up_relu = nn.ReLU(inplace=True)
        up_norm = nn.BatchNorm2d(outer_f)
        
        # Layers that are not innermost or outermost
        if layer_type == "default":
            up_conv = nn.ConvTranspose2d(inner_f * 2, outer_f, kernel_size=hps.unet_conv_kernel,
                              stride=hps.unet_conv_stride, padding=hps.unet_conv_pad)
            downlayers = [down_conv, down_relu, down_norm]
            uplayers = [up_conv, up_relu, up_norm]
            
            if dropout:
                layer = downlayers + [sublayer] + uplayers + [nn.Dropout(hps.unet_dropout)]
            else:
                layer = downlayers + [sublayer] + uplayers
            
        if layer_type == "outermost":
            up_conv = nn.ConvTranspose2d(inner_f * 2, outer_f, kernel_size=hps.unet_conv_kernel,
                              stride=hps.unet_conv_stride, padding=hps.unet_conv_pad)
            downlayers = [down_conv]
            uplayers = [up_conv, up_relu, nn.Tanh()]
            layer = downlayers + [sublayer] + uplayers
            
        if layer_type == "innermost":
            up_conv = nn.ConvTranspose2d(inner_f, outer_f, kernel_size=hps.unet_conv_kernel,
                              stride=hps.unet_conv_stride, padding=hps.unet_conv_pad)
            downlayers = [down_conv, down_relu]
            uplayers = [up_conv, up_relu, up_norm]
            layer = downlayers + uplayers
        
        self.layer = nn.Sequential(*layer) 
        
        
    def forward(self, input):
        
        if self.layer_type == "outermost":
            return self.layer(input)
        else:
            return torch.cat([input, self.layer(input)], 1)
        
        

class Unet(nn.Module):
    """
    U-net (O. Ronneberger, P. Fischer, and T. Brox. (2015)) is utilized for generator.

    """
    def __init__(self, hps, input_c, output_c, outer_f, depth, dropout=False):
        
        """
        Parameters:
        
            input_c (int)  : number of input channels
            output_c (int) : number of output channels
            outer_f (int)  : number of filters on the last convolution
            depth (int)    : number of unet layers
            dropout (bool) : use dropout layer?
        """
        
        super(Unet, self).__init__()
        
        # Recursive construction:
        
        # Innermost layer
        layers = UnetLayer(hps, 8*outer_f, 8*outer_f, 8*outer_f, "innermost", None)
        
        # Default layers
        for i in range(depth - 5):
            layers = UnetLayer(hps, 8*outer_f, 8*outer_f, 8*outer_f, "default", layers, dropout=dropout)
        for i in range(3):
            # Gradual reduction of number of filters
            layers = UnetLayer(hps, (8//2**(i+1))*outer_f, (8//2**i)*outer_f, (8//2**(i+1))*outer_f, "default", layers)
        
        # Outermost layer
        self.layers = UnetLayer(hps, input_c, outer_f, output_c, "outermost", layers)
        
    def forward(self, input):
        return self.layers(input)


"""
Discriminator used in training of the generator. Architecture and implementation according to [1].

[1] Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017).
Image-to-image translation with conditional adversarial networks.
In Proceedings of the IEEE conference on computer vision and pattern recognition
(pp. 1125-1134).
"""
class Discriminator(nn.Module):
    
    def __init__(self, input_c, input_f=64):
        super(Discriminator, self).__init__()

        padding = 1
        stride = 2
        ks = 4 # kernel size
        relu_slope = 0.2
        nof_layers = 3
        
        """
        Initialize a patchGAN discriminator
        with the following architecture:
            - C64-C128-C256-C512-output
        """
        # input layer, C64
        sequence = [nn.Conv2d(input_c, input_f, kernel_size=ks, stride=stride, padding=padding), nn.LeakyReLU(0.2, True)]
        
        # first hidden layer, C128
        m = 2**1 # filter multiplier
        m_previous = 1
        sequence += [
            nn.Conv2d(m_previous * input_f, m * input_f, kernel_size=ks, stride=stride),
            nn.BatchNorm2d(input_f * m), # Add normalization layer here
            nn.LeakyReLU(relu_slope, True)
        ]
        
        # second hidden layer, C256
        m_previous = m
        m = 2**2 # filter multiplier
        sequence += [
            nn.Conv2d(m_previous * input_f, m * input_f, kernel_size=ks, stride=stride),
            nn.BatchNorm2d(input_f * m), # Add normalization layer here
            nn.LeakyReLU(relu_slope, True)
        ]
        
        # third hidden layer, C512
        m_previous = m
        m = 2**nof_layers # filter multiplier
        sequence += [
            nn.Conv2d(m_previous * input_f, m * input_f, kernel_size=ks, stride=stride),
            nn.BatchNorm2d(input_f * m), # Add normalization layer here
            nn.LeakyReLU(relu_slope, True)
        ]
        
        # output layer with 1 output channel
        m_previous = m
        sequence += [nn.Conv2d(m_previous * input_f, 1, kernel_size=ks, stride=stride, padding=padding)]
        
        # model
        self.net = nn.Sequential(*sequence)
        
        
    def forward(self, input):
        return self.net(input)


