
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape - 1, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        inputs[:, :, [0, 1, 3, 4]] = inputs[:, :, [0, 1, 3, 4]] / 15000.0
        inputs[:, :, [3, 4]] = inputs[:, :, [3, 4]] - inputs[:, :, [0, 1]]

        inputs[:, :, 2] = inputs[:, :, 2] / 7500.0
        inputs[:, :, 5] = inputs[:, :, 5] / 30000.0

        inputs = th.concat([inputs[:, :, :5], inputs[:, :, 6:]], dim=-1)


        b, a, e = inputs.size() # (B, N, D)
        inputs = inputs.view(-1, e) # (B*N, D)
        x = F.relu(self.fc1(inputs), inplace=True) # (B*N, D')
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (B*N, D')
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh) # (B*N, n_actions)

        return q.view(b, a, -1), hh.view(b, a, -1)
