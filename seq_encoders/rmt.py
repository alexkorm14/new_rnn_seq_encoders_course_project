from collections import namedtuple
from functools import wraps, partial
from packaging import version
from itertools import zip_longest
from contextlib import nullcontext
from typing import Optional, List, Tuple
import math
import random
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder

import warnings
warnings.filterwarnings('ignore')

# constants
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
Linear = partial(nn.Linear, bias = False)

# helpers
def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def identity(t, *args, **kwargs):
    return t

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def divisible_by(numer, denom):
    return (numer % denom) == 0

# sampling helpers
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def token_shift_fn(t, ps):
    read_mem, t, write_mem = unpack(t, ps, 'b * d')
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    t = torch.cat((t, t_shift), dim = -1)
    return torch.cat((read_mem, t, write_mem), dim = -2)

def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)

# main Attend class
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if exists(mask):
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')

            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out

# positional embedding
class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 use_start_random_shift=True,
                 max_len=5000,
                 ):
        super(PositionalEncoding, self).__init__()
        self.use_start_random_shift = use_start_random_shift
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.max_len - T)
        else:
            start_pos = 0
        x = x + self.pe[:, start_pos:start_pos + T]
        return x

# rotary embedding
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# norms
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        Linear(dim, dim_inner * 2, bias = False),
        GEGLU(),
        RMSNorm(dim_inner),
        nn.Dropout(dropout),
        Linear(dim_inner, dim, bias = False)
    )

# attention
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)

    def forward(
        self,
        x,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        mask = None,
        xl_memories = None
    ):
        h = self.heads

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # add a null key / value
        # to protect against an entirely masked out sequence
        # as well as giving attention ability to attend to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        # manage memories

        next_xl_memories = torch.stack((k, v))

        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)

        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), next_xl_memories

# transformer
class ReccurentMemoryTransformerEncoder(AbsSeqEncoder):
    """Used https://github.com/lucidrains/recurrent-memory-transformer-pytorch"""
    def __init__(self,
        input_size,
        num_memory_tokens=128,
        n_layers=6,
        max_seq_len = 5000,
        causal = True,        
        dim_hidden = 64,
        n_heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_flash_attn = False,
        ignore_index = -1,
        abs_pos_emb = True,
        rotary_pos_emb = False,
        token_shift = True,
        emb_gradient_frac = 0.1,             # trick from cogview paper that leads to a bit more stability
        memory_not_causal = True,            # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
        add_write_to_next_write_mem = False, # add the write memories of previous step to the next write step - thanks to @IcarusWizard for pointing out this discrepancy
        next_write_mem_stop_grad = True,     # whether to stop gradient of previous read memory -> next write memory
        always_have_read_memories = True,    # whether to always have read memories, even on the first step, so to make the model onnx-able
        resi_dual_scale = 1.,                # in the case of overflows in fp16 on the prenorm branch, set this to a value less than 1.
        is_reduce_sequence=True,
    ):
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.causal = causal
        self.emb_gradient_frac = emb_gradient_frac
        assert 0 < resi_dual_scale <= 1., 'resiDual scale must be between 0 and 1'
        self.resi_dual_scale = resi_dual_scale
        assert num_memory_tokens > 0

        # positions
        assert any([abs_pos_emb, rotary_pos_emb, token_shift]), 'Should set True at least one of abs_pos_emb,rotary_pos_emb,token_shift'
        
        if abs_pos_emb:
            self.pos_emb = PositionalEncoding(
                use_start_random_shift=False,
                max_len=max_seq_len,
                d_model=input_size,
            )
        else:
            self.pos_emb = None

        self.rotary_pos_emb = RotaryEmbedding(dim_hidden) if rotary_pos_emb else None
        self.maybe_token_shift = token_shift_fn if token_shift else identity

        # memory related
        self.num_memory_tokens = num_memory_tokens
        self.read_memory_emb = nn.Parameter(torch.zeros(num_memory_tokens, input_size))
        nn.init.normal_(self.read_memory_emb, std = 0.02)
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, input_size))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = input_size,
                    dim_head = dim_hidden,
                    causal = causal,
                    heads = n_heads,
                    use_flash_attn = use_flash_attn,
                    use_custom_causal_attn_mask = memory_not_causal,
                    dropout = attn_dropout
                ),
                RMSNorm(input_size),
                FeedForward(dim = input_size, mult = ff_mult, dropout = ff_dropout),
                RMSNorm(input_size)
            ]))
        self.norm = RMSNorm(input_size)

        self.ignore_index = ignore_index
        # whether to use custom attention mask if causal and memory should not be causal
        self.use_custom_causal_attn_mask = causal and memory_not_causal
        # in the paper, they actually also use the previous write memories for the next write memories
        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad
        # allow for attending to raw read memory positional embeddings on first step
        # hack to make it onnx-able and should not hurt
        self.always_have_read_memories = always_have_read_memories

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

    def forward(self, 
        x: PaddedBatch,
        read_memories = None,
        *,
        mask = None,
        labels = None,
        mask_out_read_memories = False   # in the case one is passing in 0s for read memories, for onnx-able model
    ):

        x_in = x.payload
        b, n, h = x_in.size()
        device, mem_length = x_in.device, self.num_memory_tokens
        pos = torch.arange(n, device = device)

        # maybe absolute positional embedding
        if exists(self.pos_emb):
            x_in = self.pos_emb(x_in)
        
        # trick from cogview paper
        x_in = frac_gradient(x_in, self.emb_gradient_frac)
        
        # prepare write memories, as in paper
        write_memories = self.init_memory(b)

        if exists(read_memories) and self.add_write_to_next_write_mem:
            maybe_detach = torch.detach if self.next_write_mem_stop_grad else identity
            write_memories = write_memories + maybe_detach(read_memories)

        # prepare read memories
        if exists(read_memories):
            if read_memories.ndim == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b = b)
            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb

        elif self.always_have_read_memories:
            read_mem_length = mem_length
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = b)
        else:
            read_mem_length = 0
            read_memories = x_in[:, 0:0]

        # concat to main sequence using einop's pack

        x_in, ps = pack([read_memories, x_in, write_memories], 'b * d')

        # take care of mask
        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value = True)

        # custom causal mask, if needed
        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).tril()

            causal_mask = F.pad(causal_mask, (0, mem_length, read_mem_length, 0), value = False)
            causal_mask = F.pad(causal_mask, (read_mem_length, 0, 0, mem_length), value = True)

            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask & causal_mask
            else:
                mask = causal_mask

        # masking out read memories, either for passing in 0s for read memories on first step, or if you are doing some regularization game on the memories
        if read_mem_length > 0 and mask_out_read_memories:
            read_mem_mask = torch.arange(x.shape[-2], device = device) < read_mem_length

            if exists(mask):
                mask = mask & ~read_mem_mask
            else:
                mask = read_mem_mask

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0
        rotary_emb = None

        if exists(self.rotary_pos_emb):
            mem_rel_dist = 10000

            q_pos = pos + mem_rel_dist

            q_pos = F.pad(q_pos, (read_mem_length, mem_length), value = 0)
            q_rotary_emb = self.rotary_pos_emb(q_pos)

            # kind of confusing at the moment
            # but the order of the keys are - [xl memories] [read memories] [main sequence] [ write memories]
            # so the positions are (say xl memory length of 256) - [10001, 10002, 10003 ...] [0, 0, ...] [10256, 10257, ...] [0, 0, ...]
            k_pos = q_pos
            
            # account for null key / value

            k_pos = F.pad(k_pos, (1, 0), value = mem_rel_dist - 1) # give a null memory token, to allow for attending to nothing

            k_rotary_emb = self.rotary_pos_emb(k_pos)

            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # maybe token shift function
        shift_fn = partial(self.maybe_token_shift, ps = ps)

        # attention and feedforward
        residual = x_in * self.resi_dual_scale

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            
            attn_out, xl_memories = attn(shift_fn(x_in), mask = mask, xl_memories = None, rotary_emb = rotary_emb)
            
            x_in = attn_post_norm(x_in + attn_out)

            residual = residual + attn_out * self.resi_dual_scale

            ff_out = ff(shift_fn(x_in))

            x_in = ff_post_norm(x_in + ff_out)

            residual = residual + ff_out * self.resi_dual_scale

        # add final norm of residual, as in resiDual paper
        out = x_in + self.norm(residual)

        # split out memories using unpack

        read_memories, out, write_memories = unpack(out, ps, 'b * d')

        if self.is_reduce_sequence:
            return out[:, 0, :]

        return PaddedBatch(out, x.seq_lens)

    @property
    def embedding_size(self):
        return self.input_size