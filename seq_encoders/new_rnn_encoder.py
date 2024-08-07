import torch
import warnings
from torch import nn as nn
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_step import LastStepEncoder, LastMaxAvgEncoder, FirstStepEncoder
from ptls.data_load.padded_batch import PaddedBatch
from s5 import S5, S5Block
from lmu import LMU
from qrnn import QRNN
from urnn import URNN
from cornn import CoRNN
from indrnn import IndRNN
from lem import LEM
from lru import LRU

class NewRnnEncoder(AbsSeqEncoder):
    """Use torch recurrent layer network
    Based on `torch.nn.GRU` and `torch.nn.LSTM` and QRNN,LMU,CoRnn, Urnn
    Parameters
        input_size:
            input embedding size
        hidden_size:
            intermediate and output layer size
        type:
            'gru' or 'lstm' or 'qrnn' or 'lmu' or 'urnn' or 'cornn' or 'xlstm: mlstm, slstm'
        bidir:
            Bidirectional RNN
        dropout:
            RNN dropout
        trainable_starter:
            'static' - use random learnable vector for rnn starter
            other values - use None as starter
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token
    """
    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 type='gru',
                 bidir=False,
                 num_layers=1,
                 dropout=0,
                 trainable_starter='static',
                 is_reduce_sequence=True,
                 reducer='last_step',
                 **kwargs
                 ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size
        self.rnn_type = type
        self.bidirectional = bidir
        self.num_layers = num_layers
        if self.bidirectional:
            warnings.warn("Backward direction in bidir RNN takes into account paddings at the end of sequences!")

        self.trainable_starter = trainable_starter

        # initialize RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=dropout)
        elif self.rnn_type == 'mlstm':
            self.rnn = mLSTM(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                )
        elif self.rnn_type == 'slstm':
            self.rnn = sLSTM(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                )
        elif self.rnn_type == 'xlstm':
            self.rnn = xLSTM(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                dropout=dropout,
                **kwargs)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=dropout)
        elif self.rnn_type == 'qrnn':
            self.rnn = QRNN(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                dropout=dropout,
                **kwargs)
        elif self.rnn_type == 'lmu':
            self.rnn = LMU(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                **kwargs)
        elif self.rnn_type == 'cornn':
            self.rnn = CoRNN(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                **kwargs)
        elif self.rnn_type == 'urnn':
            self.rnn = URNN(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional)
        elif self.rnn_type == 'indrnn':
            self.rnn = IndRNN(
                input_size=input_size, 
                hidden_size=self.hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                bidirectional=self.bidirectional,
                **kwargs)    
        elif self.rnn_type == 'lru':
            self.rnn = LRU(
                in_features=input_size,
                out_features=self.hidden_size,
                **kwargs)
        elif self.rnn_type == 'lem':
            self.rnn = LEM(
                ninp=input_size,
                nhid=self.hidden_size,
                dropout=dropout,
                **kwargs)
        elif self.rnn_type == 's5':
            params = kwargs.copy()
            if kwargs.get('block',False):
                self.rnn = S5Block(
                    dim=input_size,
                    state_dim=self.hidden_size,
                    bidir=self.bidirectional,
                    block_count=num_layers,
                    **params)
            else:
                self.rnn = S5(
                    width=input_size,
                    state_width=self.hidden_size,
                    block_count=num_layers,
                    bidir=self.bidirectional,
                    **params)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        self.full_hidden_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2

        # initialize starter position if needed
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(self.num_layers * num_dir, 1, self.hidden_size))

        if reducer == 'last_step':
            self.reducer = LastStepEncoder()
        elif reducer == 'first_step':
            self.reducer = FirstStepEncoder()
        elif reducer == 'last_max_avg':
            self.reducer = LastMaxAvgEncoder()

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """
        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        assert shape[1] > 0, "Batch can'not have 0 transactions"

        # prepare initial state
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            starter_h = torch.tanh(self.starter_h.expand(self.num_layers * num_dir, shape[0], -1).contiguous())
            if h_0 is None:
                h_0 = starter_h
            elif h_0 is not None and not self.training:
                h_0 = torch.where(
                    (h_0.squeeze(0).abs().sum(dim=1) == 0.0).unsqueeze(0).unsqueeze(2).expand(*starter_h.size()),
                    starter_h,
                    h_0,
                )
            else:
                raise NotImplementedError('Unsupported mode: cannot mix fixed X and learning Starter')
        # pass-through rnn
        if self.rnn_type == 'gru':
            out, _ = self.rnn(x.payload, h_0)
        elif self.rnn_type in ['lstm','slstm','mlstm','xlstm','urnn','cornn','qrnn','lmu','indrnn']:
            out, _ = self.rnn(x.payload)
        elif self.rnn_type in ['lru','lem','s5']:
            out = self.rnn(x.payload)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            return self.reducer(out)
        return out

    @property
    def embedding_size(self):
        return self.hidden_size