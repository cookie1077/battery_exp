"""
Battery Projects Models
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.activation = nn.ReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, self.sequence_length)
        x = torch.mean(x, axis=1, keepdim=True)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss_function(y, y_hat)
        return loss

class RNN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNN_GRU, self).__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, output_dim, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc1(x[:,-1])
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss_function(y, y_hat)


class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNN_LSTM, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, output_dim, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc1(x[:,-1])
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss_function(y, y_hat)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout):
        super(BidirectionalLSTM, self).__init__()
        """
        Args:
            nIn (int): The number of input unit
            nHidden (int): The number of hidden unit
            nOut (int): The number of output unit
        """
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=False, batch_first=True)
        self.embedding = nn.Linear(nHidden, nOut)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = dropout

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(b * T, h)

        if self.dropout:
            t_rec = self.dropout(t_rec)
        output = self.embedding(t_rec)
        output = output.contiguous().view(b, T, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, ni, nc, no, nh, n_rnn=2, leakyRelu=False, sigmoid=False):
        """
        Args:
            ni (int): The number of input unit
            nc (int): The number of original channel
            no (int): The number of output unit
            nh (int): The number of hidden unit
        """
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [0, 0, 0, 0, 0, 0, 0]
        ss = [2, 2, 2, 2, 2, 2, 1]
        nm = [8, 16, 64, 64, 64, 64, 64]

        cnn = nn.Sequential()

        def convRelu(i, cnn, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            if i == 3: nIn = 64
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, (ks[i], 1), (ss[i], 1), (ps[i], 0)))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, cnn)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 1), (2, 1)))
        convRelu(1, cnn)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 1), (2, 1)))
        convRelu(2, cnn)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 1), (2, 1), (0, 0)))
        self.sigmoid = sigmoid
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(64, nh, nh, False),
            BidirectionalLSTM(nh, nh, no, False), )
        self.rul = nn.Linear(10, 1)
        self.soh = nn.Linear(64, 1)

    def forward(self, input):
        """
        Input shape: [b, c, h, w]
        Output shape:
            rul [b, 1]
            soh [b, 10]
        """
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        output = self.rnn(conv)
        soh = self.soh(output).squeeze()

        if not self.sigmoid:
            rul = self.rul(soh)
        else:
            rul = F.sigmoid(self.rul(soh))

        return rul, soh


class FCModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction="mean")
        self.activation = nn.Tanh()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias)

    def forward(self, x):
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss(self, x, y):
        g = self.forward(x)
        loss = self.loss_function(g, y)
        return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.init_weights()
        self.loss_function = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[:, -1, :]  # 마지막 시퀀스 요소만 반환

    def loss(self, x, y):
        y_hat = self.forward(x)
        mse_loss = self.loss_function(y, y_hat)

        return mse_loss


class LTSF_Linear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_Linear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        self.loss_function = nn.MSELoss()

        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x[:, :, 0]

    def loss(self, x, y):
        y_hat = self.forward(x)
        mse_loss = self.loss_function(y, y_hat)

        return mse_loss

class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual


class LTSF_DLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size):
        super(LTSF_DLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Trend.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))

    def forward(self, x):
        trend_init, seasonal_init = self.decompsition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forcast_size],
                                       dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forcast_size],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x