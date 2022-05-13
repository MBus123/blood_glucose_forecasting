
import torch.nn as nn
import torch


class Block(nn.Module):
    """
    A single block of the n-BEATS model
    """

    def __init__(self, input_dim=24, parameter_dim=12, output_dim=6, amount_fc=3, hidden_dim=128):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.parameter_dim = parameter_dim
        self.output_dim = output_dim
        linear_layers = [nn.Linear(input_dim * parameter_dim, hidden_dim), nn.ReLU()]
        for _ in range(amount_fc - 2):
            linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(hidden_dim, input_dim + output_dim))
        self.layers = nn.Sequential(
            *linear_layers
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class NBeats(nn.Module):
    """
    The plain N-BEATS model implementation
    """

    def __init__(self, n_blocks=12, input_dim=24, parameter_dim=12, output_dim=6, amount_fc=3, hidden_dim=128):
        """
        Constructs the model.

        Parameters
        ----------
        n_blocks : int
            the amount of blocks used for the modell
        input_dim : int
            the amount of past time stepts used for input
        parameter_dim : int
            The amount of parameters used for each time step
        output_dim : int
            The prediction horizon
        hidden_dim : 
            The size of the hidden layers inside a single block
        """
        super(NBeats, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.parameter_dim = parameter_dim
        self.loss_function = nn.MSELoss()
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(Block(input_dim=input_dim, parameter_dim=parameter_dim, output_dim=output_dim, amount_fc=amount_fc, hidden_dim=hidden_dim))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        inputs = [x]
        self.x_array = []
        self.y_array = []
        device = "cuda:0" if x.is_cuda else "cpu"
        self.y = torch.zeros(x.shape[0], self.output_dim, device=device)
        for block in self.blocks:
            x1 = block(inputs[-1])
            x_backcast, x_forecast = torch.split(x1, (self.input_dim, self.output_dim), dim=1)
            self.y += x_forecast
            self.y_array.append(self.y.clone())
            self.x_array.append((x[:, :self.input_dim].clone(), x_backcast[:, :self.input_dim].clone()))
            x_tmp = x.clone()
            x_tmp[:, :self.input_dim] -= x_backcast
            inputs.append(x_tmp)

        return self.y

    def calculate_loss(self, y):
        """
        Calculates the custom loss described in: http://ceur-ws.org/Vol-2675/paper18.pdf
        
        """
        backcast_loss = 0
        forecast_loss = 0
        magnitude_loss = 0
        for i, (x1, x2) in enumerate(self.x_array, 1):
            backcast_loss += i * self.loss_function(x2, x1)
            magnitude_loss += (1 / i) * (1 / x2.abs().sum())
        backcast_loss /= sum([i for i in range(1, len(self.blocks) + 1)])
        magnitude_loss /= sum([1 / i for i in range(1, len(self.blocks) + 1)])
        for i, y_pred in enumerate(self.y_array, 1):
            forecast_loss += i ** 3 * self.loss_function(y_pred, y)
        forecast_loss /= sum([i for i in range(1, len(self.blocks) + 1)])

        return forecast_loss +  0.6 * backcast_loss +  0.4 * magnitude_loss

