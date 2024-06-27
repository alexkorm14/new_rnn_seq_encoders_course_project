# CoRNN - https://github.com/tk-rusch/coRNN/tree/master
import torch
from torch import nn
from torch.autograd import Variable

class CoRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dt, gamma, epsilon):
        super(CoRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1))) - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class CoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=5.4e-2, gamma=4.9, epsilon=4.8,
                 num_layers=1, bidirectional=False):
        super(CoRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(CoRNNCell(input_size,hidden_size,dt,gamma,epsilon))
        for l in range(1,self.num_layers):
            self.rnn_cell_list.append(CoRNNCell(hidden_size,hidden_size,dt,gamma,epsilon))


    def forward(self, input, state=None):

        # Input of shape (batch_size, sequence length, input_size)

        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = input.size(0)
        seq_len = input.size(1)

        # Initial state
        if state == None:
            hy = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
            hz = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)

            if self.bidirectional:
                hy_b = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
                hz_b = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
              hidden_forward.append(torch.stack([hy[layer, :, :], hz[layer, :, :]]))

        if self.bidirectional:
            hidden_backward = list()
            for layer in range(self.num_layers):
                  hidden_backward.append(torch.stack([hy_b[layer, :, :], hz_b[layer, :, :]]))

        # Iterate over the sequence
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    # Forward net
                    h_forward_y,h_forward_z = self.rnn_cell_list[layer](input[:, t, :], hidden_forward[layer][0],hidden_forward[layer][1])
                    if self.bidirectional:
                        # Backward net
                        h_back_y,h_back_z = self.rnn_cell_list[layer](input[:, -(t + 1), :], hidden_backward[layer][0],hidden_backward[layer][1])
                else:
                    # Forward net
                    h_forward_y,h_forward_z = self.rnn_cell_list[layer](hidden_forward[layer - 1][0], hidden_forward[layer][0],hidden_forward[layer][1])
                    if self.bidirectional:
                        # Backward net
                        h_back_y,h_back_z = self.rnn_cell_list[layer](hidden_backward[layer - 1][0], hidden_backward[layer][0],hidden_backward[layer][1])

                hidden_forward[layer] = torch.stack([h_forward_y,h_forward_z])

                if self.bidirectional:
                    hidden_backward[layer] = torch.stack([h_back_y,h_back_z])

            outs.append(torch.stack(hidden_forward)[-1,0])
            if self.bidirectional:
                outs_rev.append(torch.stack(hidden_backward)[-1,0])

        outs = torch.stack(outs)
        if self.bidirectional:
            outs_rev = torch.stack(outs_rev)
            outs = torch.cat((outs, outs_rev),2)
            h_out = torch.cat((torch.stack(hidden_forward)[:,0],torch.stack(hidden_backward)[:,0]))
        else:
            h_out = torch.stack(hidden_forward)[:,0]

        return outs.permute(1,0,2), h_out