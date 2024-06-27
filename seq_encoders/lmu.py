# LMU - https://github.com/hrshtv/pytorch-lmu
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.signal import cont2discrete

def leCunUniform(tensor):
    """
        LeCun Uniform Initializer
        References:
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)

class LMUCell(nn.Module):
    """
    LMU Cell

    Parameters:
        input_size (int) :
            Size of the input vector (x_t)
        hidden_size (int) :
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):

        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)

        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters()

    def initParameters(self):
        """ Initialize the cell's parameters """

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        init.constant_(self.e_m, 0)
        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m)

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")
        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))
        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D),
            dt = 1.0,
            method = "zoh"
        )

        return A, B

    def forward(self, x, h, m):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, input_size]
            state (tuple):
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """
        # Equation (7) of the paper
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m) # [batch_size, 1]
        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B) # [batch_size, memory_size]
        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) +
            F.linear(m, self.W_m)
        ) # [batch_size, hidden_size]

        return h, m

class LMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = False, learn_b= False,
                 num_layers=1, bidirectional=False):
        super(LMU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b))
        for l in range(1,self.num_layers):
            self.rnn_cell_list.append(LMUCell(hidden_size, hidden_size, memory_size, theta, learn_a, learn_b))


    def forward(self, input, state=None):

        # Input of shape (batch_size, sequence length, input_size)

        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = input.size(0)
        seq_len = input.size(1)

        # Initial state
        if state == None:
            h = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
            m = Variable(torch.zeros(self.num_layers, batch_size, self.memory_size)).to(self.device)

            if self.bidirectional:
                h_b = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
                m_b = Variable(torch.zeros(self.num_layers, batch_size, self.memory_size)).to(self.device)

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
              hidden_forward.append([h[layer, :, :], m[layer, :, :]])

        if self.bidirectional:
            hidden_backward = list()
            for layer in range(self.num_layers):
                  hidden_backward.append([h_b[layer, :, :], m_b[layer, :, :]])

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

                hidden_forward[layer] = [h_forward_y,h_forward_z]

                if self.bidirectional:
                    hidden_backward[layer] = [h_back_y,h_back_z]

            outs.append(hidden_forward[-1][0])
            if self.bidirectional:
                outs_rev.append(hidden_backward[-1][0])

        outs = torch.stack(outs)
        if self.bidirectional:
            outs_rev = torch.stack(outs_rev)
            outs = torch.cat((outs, outs_rev),2)
            h_out = torch.cat((torch.stack([hf[0] for hf in hidden_forward]),torch.stack([hb[0] for hb in hidden_backward])))
        else:
            h_out = torch.stack([hf[0] for hf in hidden_forward])

        return outs.permute(1,0,2), h_out