# URnn - https://github.com/rand0musername/urnn
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

# Diagonal unitary matrix
class DiagonalMatrix(nn.Module):
    def __init__(self, num_units):
        super(DiagonalMatrix, self).__init__()
        self.w = Variable(init.uniform_(torch.empty(num_units),a=-np.pi, b=np.pi),requires_grad=True).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.vec = torch.complex(torch.cos(self.w), torch.sin(self.w)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # [batch_sz, num_units]
    def mul(self, z):
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return self.vec * z

# Reflection unitary matrix
class ReflectionMatrix(nn.Module):
    def __init__(self, num_units):
        super(ReflectionMatrix, self).__init__()
        self.num_units = num_units
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.re = Variable(init.uniform_(torch.empty(num_units),a=-1, b=1),requires_grad=True).to(self.device)
        self.im = Variable(init.uniform_(torch.empty(num_units),a=-1, b=1),requires_grad=True).to(self.device)
        self.v = torch.complex(self.re, self.im).to(self.device) # [num_units]
        self.vstar = torch.conj(self.v).to(self.device) # [num_units]

    # [batch_sz, num_units]
    def mul(self, z):
        v = torch.unsqueeze(self.v, 1) # [num_units, 1]
        vstar = torch.conj(v) # [num_units, 1]
        vstar_z = torch.matmul(z, vstar) #[batch_size, 1]
        sq_norm = torch.sum(torch.abs(self.v)**2) # [1]
        factor = (2 / torch.complex(sq_norm, torch.tensor(0, dtype=torch.float32).to(self.device)).to(self.device))
        return z - factor * torch.matmul(vstar_z, v.T)

# Permutation unitary matrix
class PermutationMatrix(nn.Module):
    def __init__(self, num_units):
        super(PermutationMatrix, self).__init__()
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.P = torch.from_numpy(perm).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # [batch_sz, num_units], permute columns
    def mul(self, z):
        return z.T[self.P].T

def modReLU(z, bias):
    # relu(|z|+b) * (z / |z|)
    norm = torch.abs(z)
    scale = F.relu(norm + bias) / (norm + 1e-6)
    scaled = torch.complex(torch.real(z)*scale, torch.imag(z)*scale)
    return scaled

class URNNCell(nn.Module):
    """The most basic URNN cell.
    Args:
        hidden_size (int): hidden layer size.
        input_size: input layer size.
    """
    def __init__(self, input_size, hidden_size):
        super(URNNCell, self).__init__()
        # save class variables
        self._num_in = input_size
        self._num_units = hidden_size
        self._state_size = hidden_size*2
        self._output_size = hidden_size*2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up input -> hidden connection

        self.w_ih = Variable(init.xavier_uniform_(torch.empty([2*hidden_size, input_size])),requires_grad=True).to(self.device)
        self.b_h = Variable(torch.zeros(hidden_size),requires_grad=True).to(self.device) # state size actually

        # elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix(hidden_size)
        self.R1 = ReflectionMatrix(hidden_size)
        self.D2 = DiagonalMatrix(hidden_size)
        self.R2 = ReflectionMatrix(hidden_size)
        self.D3 = DiagonalMatrix(hidden_size)
        self.P = PermutationMatrix(hidden_size)

    def forward(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """
        # prepare input linear combination
        inputs_mul = torch.matmul(inputs, self.w_ih.T) # [batch_sz, 2*num_units]

        inputs_mul_c = torch.complex(inputs_mul[:, :self._num_units], inputs_mul[:, self._num_units:]).to(self.device)
        # [batch_sz, num_units]
        # prepare state linear combination (always complex!)
        state_c = torch.complex(state[:, :self._num_units], state[:, self._num_units:]).to(self.device)

        state_mul = self.D1.mul(state_c)
        state_mul = torch.fft.fft(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
        state_mul = self.D2.mul(state_mul)
        state_mul = torch.fft.ifft(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul)
        # [batch_sz, num_units]
        # calculate preactivation
        preact = inputs_mul_c + state_mul
        # [batch_sz, num_units]
        new_state_c = modReLU(inputs_mul_c, self.b_h).to(self.device) # [batch_sz, num_units] C
        new_state = torch.concat([torch.real(new_state_c), torch.imag(new_state_c)], 1).to(self.device) # [batch_sz, 2*num_units] R
          
        return new_state

class URNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(URNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        assert num_layers == 1, 'For URNN num_layer should be equal 1'
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(URNNCell(input_size,hidden_size))

    def forward(self, input, state=None):

        # Input of shape (batch_size, sequence length, input_size)
        
        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = input.size(0)
        seq_len = input.size(1)

        # Initial state
        if state == None:
            gap = np.sqrt(3 / (2 * self.hidden_size))
            h = Variable(init.uniform_(torch.empty(self.num_layers, batch_size, 2 * self.hidden_size),a=-gap, b=gap)).to(self.device)
            
            if self.bidirectional:
                h_b = Variable(init.uniform_(torch.empty(self.num_layers, batch_size, 2 * self.hidden_size),a=-gap, b=gap)).to(self.device)
                
        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
              hidden_forward.append(h[layer, :, :])

        if self.bidirectional:
            hidden_backward = list()
            for layer in range(self.num_layers):
                  hidden_backward.append(h_b[layer, :, :])

        # Iterate over the sequence
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    # Forward net
                    h_forward = self.rnn_cell_list[layer](input[:, t, :], hidden_forward[layer])
                    if self.bidirectional: 
                        # Backward net
                        h_back = self.rnn_cell_list[layer](input[:, -(t + 1), :], hidden_backward[layer])
                else:
                    # Forward net
                    h_forward = self.rnn_cell_list[layer](hidden_forward[layer - 1], hidden_forward[layer])
                    if self.bidirectional: 
                        # Backward net
                        h_back = self.rnn_cell_list[layer](hidden_backward[layer - 1], hidden_backward[layer])

                hidden_forward[layer] = h_forward
  
                if self.bidirectional:
                    hidden_backward[layer] = h_back
       
            outs.append(torch.stack(hidden_forward)[-1])
            if self.bidirectional:
                outs_rev.append(torch.stack(hidden_backward)[-1])

        outs = torch.stack(outs)
        if self.bidirectional:
            outs_rev = torch.stack(outs_rev)
            outs = torch.cat((outs, outs_rev),2)
            h_out = torch.cat((torch.stack(hidden_forward),torch.stack(hidden_backward)))
        else:
            h_out = torch.stack(hidden_forward)

        return outs.permute(1,0,2), h_out