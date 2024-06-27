# QRNN - https://github.com/salesforce/pytorch-qrnn/tree/master/torchqrnn

import torch
from torch.autograd import Variable

class ForgetMult(torch.nn.Module):
    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: 
                h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h

        return torch.stack(result)


class QRNNLayer(torch.nn.Module):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.

    Inputs: X, hidden
        - X (batch, seq_len, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size, dropout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()
        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"

        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_gate = output_gate
        self.forget_mult = ForgetMult()
        
        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = torch.nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def forward(self, X, hidden=None):
        X = X.permute(1,0,2)
        seq_len, batch_size, _ = X.size()
        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)

        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.dropout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.dropout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.dropout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = self.forget_mult(F, Z, hidden)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        return H, C[-1,:, :]


class QRNN(torch.nn.Module):
    """Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        dropout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers, batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (batch, seq_len, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """
    def __init__(self, input_size, hidden_size, window=1, output_gate=True,
                 num_layers=1, bidirectional=False, dropout=0):
        super(QRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.rnn_cell_list = torch.nn.ModuleList()
        self.rnn_cell_list.append(QRNNLayer(input_size, hidden_size, dropout=dropout, window=window, output_gate=output_gate))
        for l in range(1,self.num_layers):
            self.rnn_cell_list.append(QRNNLayer(hidden_size, hidden_size, dropout=dropout, window=window, output_gate=output_gate))

    def forward(self, input, state=None):

        # Input of shape (batch_size, sequence length, input_size)
        
        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = input.size(0)
        seq_len = input.size(1)

        input_f = input
        hidden_forward = list()
        if self.bidirectional:
            input_b = torch.flip(input, [1])
            hidden_backward = list()

        # Iterate over layers
        for layer in range(self.num_layers):
            # Forward net
            input_f, h_forward = self.rnn_cell_list[layer](input_f)
            if self.bidirectional: 
                # Backward net
                input_b, h_back = self.rnn_cell_list[layer](input_b)            
            
            hidden_forward.append(h_forward)
            if self.bidirectional:
                hidden_backward.append(h_back)

            if self.dropout != 0 and layer < len(self.rnn_cell_list) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)
    
        if self.bidirectional:
            outs = torch.cat((input_f, input_b),2)
            h_out = torch.cat((torch.stack(hidden_forward),torch.stack(hidden_backward)))
        else:
            h_out = torch.stack(hidden_forward)
            outs = input_f

        return outs.permute(1,0,2), h_out