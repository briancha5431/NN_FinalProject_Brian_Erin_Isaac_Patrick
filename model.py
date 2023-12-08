import torch
import torch.nn as nn

architecture_config = [
    # This architecture of the YOLO baseline is derived from the YOLO paper by Joseph Redmon.
    # Source: https://arxiv.org/pdf/1506.02640v5.pdf
    # Page 3, Figure 3.
    
    # For tuples: (kernel_size, number_filters, stride, padding)
    # Padding is calculated by hand using the final dimensions of the output
    
    # Convolutional layer with a kernel size of 7x7, 64 output channels (filters), a stride of 2, and padding of 3
    (7, 64, 2, 3),
    
    "M",    # Max-pooling layer with a 2x2 kernel and a stride of 2.
    
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    
    # Series of convolutional layers
    # 1x1 convolution with 256 output channels, 1 stride, and no padding
    # 3x3 convolution with 512 output channels, 1 stride, and padding of 1
    # Last value represents number of repeats (i.e. 4 in this case)
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    
    # Does not include fully connected layers
]


class CNNBlock(nn.Module):
    '''
        This class represents each convolutional layer (block) that is designed to 
        implement our YOLO V1 model. 
        
        The CNN Block typically consists of convolutional layers, activation 
        functions, pooling layers, and optionally normalization layers
    '''
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        # Bias=False is necessary to perform batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)  # batch normalization
        self.leakyrelu = nn.LeakyReLU(0.1)             # Slope was 0.1 as defined in the paper
        
    def forward(self, x):
        # Forward pass requires each sequential layer (conv -> batchnorm -> activation function)
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    '''
        Our Yolo V1 model as specified in the paper.
    '''

    def __init__(self, in_channels=3, **kwargs):        # in channels default to 3 to take in rgb images
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels

        # Darknet: fully connected convolutional layers system
        # open-source neural network framework (written by Redmon)
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)  # fully connected
        
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))      # starting dim as 1 prevents flattening examples
    
    def _create_conv_layers(self, architecture):
        '''
            This function creates a custom DarkNet (series of convolutional layers).
        '''

        layers = []
        in_channels = self.in_channels
        
        # x is a "step" in the architecture
        for x in architecture:
            if type(x) == tuple:        # convolutional layer
                layers += [CNNBlock(
                    in_channels, 
                    out_channels=x[1], 
                    kernel_size=x[0], 
                    stride=x[2], 
                    padding=x[3]
                )]

                # Output of previous layer is the input to the next layer
                in_channels = x[1]
            
            elif type(x) == str:        # max pooling aka the "M"
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            elif type(x) == list:       # multiple convolutional layers
                conv1 = x[0]            # layer 1
                conv2 = x[1]            # layer 2
                num_repeats = x[2]      # repeat integer
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )]
                    
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )]
                    
                    # Output of previous layer is the input to the next layer
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        '''
            Fully connected layer
        '''
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),                           # Flatten to send to linear layer
            nn.Linear(1024 * S * S, 496),           # Original Paper had 4096 as the output dim
                                                    # Using 496 to save memory
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),    # Reshaped to (S, S, 30), where C + B * 5 = 30
        )
    


def test(S=7, B=2, C=20):
    '''
        Simple test case to see if our model is configured correctly.
    '''

    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape == torch.Size([2, 1470]))

test() # torch.Size([2, 1470])

# This file is done! test passed
