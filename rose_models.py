from torch import nn

# Variable Generator
class Generator(nn.Module):
    def __init__(self, layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1), output_activation=nn.Tanh()):
        """A generator for mapping a latent space to a sample space.
        Args:
            layers_size (List[int]): A list of layer widths, 
                e.g [input_size, hidden_size_1, ..., hidden_size_3, output_size]
            layer_activation: torch activation function to follow all layers except the output layer
            output_activation: torch activation function or None
        Example: 
            layer_size=[2,512,512,512,2] produces the following 4 layers
            (0): Linear(in_features=2, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=2, bias=True)
        """
        super(Generator, self).__init__()
        self.input_size = layer_size[0]
        self.layer_size = layer_size[1:]
        self.layers = nn.ModuleList()
        self.a = layer_activation #LeakyReLU
        self.o = output_activation #Tanh or SELU


        current_dim = self.input_size
        for layer_dim in self.layer_size:
            self.layers.append(nn.Linear(current_dim, layer_dim)) # Add all layers
            current_dim = layer_dim

        print(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.a(layer(x))
        x = self.layers[-1](x)
        return self.o(x) if self.o is not None else x

class Discriminator(nn.Module):
    def __init__(self, layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1), output_activation=nn.Sigmoid()):
        """A discriminator for mapping a latent space to a sample space.
        Args:
            layers_size (List[int]): A list of layer widths, 
                e.g [input_size, hidden_size_1, ..., hidden_size_3, output_size]
            layer_activation: torch activation function to follow all layers except the output layer
            output_activation: torch activation function or None
        """
        super(Discriminator, self).__init__()
        self.input_size = layer_size[0]
        self.layer_size = layer_size[1:]
        self.layers = nn.ModuleList()
        self.a = layer_activation #LeakyReLU
        self.o = output_activation #Sigmoid


        current_dim = self.input_size
        for layer_dim in self.layer_size:
            self.layers.append(nn.Linear(current_dim, layer_dim)) # Add all layers
            current_dim = layer_dim

        print(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.a(layer(x))
        x = self.layers[-1](x)
        return self.o(x)