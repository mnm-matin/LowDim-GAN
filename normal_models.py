from torch import nn


# 1D Generator
class Normal_Generator(nn.Module):
    def __init__(self, latent_dim):
        """A generator for the normal distribution with variable latent dimensions.
        Args:
            latent_dim (int): latent dimension
        """
        super(Normal_Generator, self).__init__()
        self.latent_dim = latent_dim

        self.map1 = nn.Linear(latent_dim, 64)
        self.map2 = nn.Linear(64, 32)
        self.map3 = nn.Linear(32, 1)
        self.a = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass to map noize z to target y"""
        x = self.map1(x)
        x = self.a(x)
        x = self.map2(x)
        x = self.a(x)
        x = self.map3(x)
        return x


# 1D Discriminator
class Normal_Discriminator(nn.Module):
    def __init__(self, input_dim):
        """A discriminator for discerning real from generated samples.
        Args:
            input_dim (int): width of the input (output of generator)
        """
        super(Normal_Discriminator, self).__init__()
        self.input_dim = input_dim

        self.map1 = nn.Linear(input_dim, 64)
        self.map2 = nn.Linear(64, 32)
        self.map3 = nn.Linear(32, 1)
        self.a = nn.LeakyReLU()
        self.f = nn.Sigmoid()


    def forward(self, input_tensor):
        """Forward pass; output confidence probability of sample being real"""
        x = self.map1(x)
        x = self.a(x)
        x = self.map2(x)
        x = self.a(x)
        x = self.map3(x)
        x = self.f(x)
        return x


# 1D Variable Generator
class V_Generator(nn.Module):
    def __init__(self, latent_dim, layers, output_activation=None):
        """A generator for mapping a latent space to a sample space.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            layers (List[int]): A list of layer widths including output width
            output_activation: torch activation function or None
        """
        super(V_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self._init_layers(layers)

    def _init_layers(self, layers):
        """Initialize the layers and store as self.module_list."""
        self.module_list = nn.ModuleList()
        last_layer = self.latent_dim
        for index, width in enumerate(layers):
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU())
        else:
            if self.output_activation is not None:
                self.module_list.append(self.output_activation())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate

# 1D Variable Discriminator
class V_Discriminator(nn.Module):
    def __init__(self, input_dim, layers):
        """A discriminator for discerning real from generated samples.
        params:
            input_dim (int): width of the input
            layers (List[int]): A list of layer widths including output width
        Output activation is Sigmoid.
        """
        super(V_Discriminator, self).__init__()
        self.input_dim = input_dim
        self._init_layers(layers)

    def _init_layers(self, layers):
        """Initialize the layers and store as self.module_list."""
        self.module_list = nn.ModuleList()
        last_layer = self.input_dim
        for index, width in enumerate(layers):
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU())
        else:
            self.module_list.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate


# 2D Generator
