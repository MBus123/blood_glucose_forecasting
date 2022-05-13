import torch.nn as nn
import torch
from torch.nn import LSTM


class LSTMEncoder(nn.Module):
    """
    Encoder, consuming the input and returns the last hidden state.
    """

    def __init__(self, input_size=11, hidden_size=10, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)


    def forward(self, x):
        # x = x.reshape((-1, x.shape[1] // self.input_size, self.input_size)).permute(1, 0, 2)
        _, (h, c) = self.lstm(x)
        return h, c



class LSTMDecoder(nn.Module):
    """
    Decoder, consuming the last hidden and cell state of the encoder and returns the forecasted values.
    """

    def __init__(self, input_size=11, output_size=1, hidden_size=10, prediction_horizon=6, num_layers=1, bidirectional=False):
        super(LSTMDecoder, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.input_size = input_size
        self.lstm = LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size) if not bidirectional else nn.Linear(hidden_size * 2, output_size)


    def forward(self, x, h, c, teacher_force=False):
        if not teacher_force:
            results = [x]
            for _ in range(self.prediction_horizon):
                out, (h, c) = self.lstm(x, (h, c))
                out = self.fc(out)
                results.append(out)
            return torch.cat(results[1:], dim=0)
            
        # shape x: seq length (pred horizon), batch size, feature length
        out, (h, c) = self.lstm(x, (h, c))
        # shape out: seq length, batch size, hidden size
        out = out.permute(1, 0, 2)
        # out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class LSTMPredictor(nn.Module):
    """
    LSTM sequence to sequence model to forecast h values, inputting T values. 
    """

    def __init__(self, input_size=11, output_size=1, hidden_size=10, num_layers=1, prediction_horizon=6, bidirectional=False):
        """
        Constructs the encoder and decoder given the parameters

        Parameters
        ----------
        input_size : int
            Per time step, how many different values are used for prediction
        output_size : int
            Per time step, how many output values should be produced
        hidden_size : int
            The hidden size of both encoder and decoder
        num_layers : int
            The amount of layers of decoder / encoder
        prediction_horizon : int
            The amount of time steps to forecast
        bidirectional : bool
            Whether to use bidirectional LSTMS
        """
        super(LSTMPredictor, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = LSTMDecoder(input_size, 1, hidden_size, prediction_horizon, num_layers, bidirectional)

    def forward(self, x, y, teacher_force=True):
        """
        Predicts the prediction horzizon

        Parameters
        ----------
        x : tensor
            input
        y : tensor
            size(batch size, 1) if teacher force is False, otherwise size(batch size, h)
        teacher_force : bool
            set to True in training, otherwise False
        """
        h, c = self.encoder(x)
        y = self.decoder(y, h, c, teacher_force)
        return y
