import torch

class SimpleSeqEncoder(torch.nn.Module):
    """Base class for Sequence encoder.
    Include `TrxEncoder` and  NewRnnEncoder implementation

    Parameters
        trx_encoder:
            TrxEncoder object
        seq_encoder:
            RnnEncoder implementation class
    """
    def __init__(self, trx_encoder, seq_encoder):
        super().__init__()

        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder

    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.trx_encoder.category_max_size

    @property
    def category_names(self):
        return self.trx_encoder.category_names

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size

    def forward(self, x, h_0=None):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x,h_0)
        return x