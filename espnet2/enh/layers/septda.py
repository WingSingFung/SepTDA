# The implementation of sepformer in SepTDA ICASSP2024

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet2.enh.layers.tcn import choose_norm

class SepTDABlock(nn.Module):
    """
    A wrapper for the implementation of the improved Sepformer block mentioned in SepTDA.
    Reference:
        BOOSTING UNKNOWN-NUMBER SPEAKER SEPARATION WITH TRANSFORMER DECODER-BASED ATTRACTOR
    Args:
        
    """
    def __init__(
        self,
        rope_intra_chunk,
        rope_inter_chunk,
        block_type, # "dual" or "triple"
        hidden_dim,
        dropout,
        activation,
        norm_type,
        # rnn setup
        rnn_type,
        rnn_dim,
        bidirectional,
        # self-attention setup
        att_heads,
        attention_dim,
        flash_attention,
        # ffn setup
        expansion_factor,
    ):
        super().__init__()
        self.intra_chunk = LSTMAttentionBlock(
            rope=rope_intra_chunk,
            hidden_dim=hidden_dim,
            norm_type=norm_type,
            dropout=dropout,
            activation=activation,
            
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            rnn_dim=rnn_dim,
            
            att_heads=att_heads,
            attention_dim=attention_dim,
            flash_attention=flash_attention,
            
            expansion_factor=expansion_factor,
        )
        self.inter_chunk = LSTMAttentionBlock(
            rope=rope_inter_chunk,
            hidden_dim=hidden_dim,
            norm_type=norm_type,
            dropout=dropout,
            activation=activation,

            rnn_type=rnn_type,
            bidirectional=bidirectional,
            rnn_dim=rnn_dim,

            att_heads=att_heads,
            attention_dim=attention_dim,
            flash_attention=flash_attention,
            
            expansion_factor=expansion_factor,
        )
        self.block_type = block_type
        if block_type == "triple":
            self.inter_spk = LSTMAttentionBlock(
                rope=None,
                hidden_dim=hidden_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation,
                
                rnn_type=None,
                bidirectional=False,
                rnn_dim=None,
                
                att_heads=att_heads,
                attention_dim=attention_dim,
                flash_attention=flash_attention,
                
                expansion_factor=expansion_factor,
            )
        self.norm_output = choose_norm(norm_type, hidden_dim)
        
    def forward(self, x):
        """
        permute -> intra_chunk -> permute -> inter_chunk -> permute -> inter_spk -> permute -> norm_output
        Args:
            x: Tensor of shape dual (batch, k, s, d) or triple (batch, c, k, s, d)
            
        Returns:
            Tensor of shape (batch, k, s, d) or (batch, c, k, s, d)
        """
        if self.block_type == "dual":
            assert len(x.shape) == 4, f"Expected 4D input, but got {len(x.shape)}D input"
            residual = x
            x = rearrange(x, "b k s d -> b s k d") # [b, k, s, d] -> [b, s, k, d]
            x = self.intra_chunk(x) # [b, s, k, d]
            x = rearrange(x, "b s k d -> b k s d") # [b, s, k, d] -> [b, k, s, d]
            x = self.inter_chunk(x) # [b, k, s, d]
            x = x + residual
            x = self.norm_output(x)
            return x
        elif self.block_type == "triple":
            assert len(x.shape) == 5, f"Expected 5D input, but got {len(x.shape)}D input"
            b, c, k, s, d = x.shape
            residual = x
            x = rearrange(x, "b c k s d -> b (c s) k d", c=c, s=s) # [b, c, k, s, d] -> [b, c*s, k, d]
            x = self.intra_chunk(x) # [b, c*s, k, d]
            x = rearrange(x, "b (c s) k d -> b (c k) s d", c=c, s=s, k=k) # [b, c*s, k, d] -> [b, c*k, s, d]
            x = self.inter_chunk(x) # [b, c*k, s, d]
            x = rearrange(x, "b (c k) s d -> b (k s) c d", c=c, k=k, s=s) # [b, c*k, s, d] -> [b, k*s, c, d]
            x = self.inter_spk(x, flatten=True) # [b, k*s, c, d]
            x = rearrange(x, "b (k s) c d -> b c k s d", c=c, k=k, s=s) # [b, k*s, c, d] -> [b, c, k, s, d]
            x = x + residual
            x = self.norm_output(x)
            return x
            

class LSTMAttentionBlock(nn.Module):
    """
    A wrapper for the implementation of the LSTM-attention block mentioned in SepTDA.
    Reference:
        BOOSTING UNKNOWN-NUMBER SPEAKER SEPARATION WITH TRANSFORMER DECODER-BASED ATTRACTOR

    Args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        hidden_dim (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        rnn_dim (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm_type (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        rope,
        # general params
        hidden_dim,    
        norm_type,
        dropout,
        activation,
        # rnn related
        rnn_type,
        bidirectional,
        rnn_dim,
        # attention related 
        att_heads,
        attention_dim,
        flash_attention,
        # feed forward related
        expansion_factor,
    ):
        super().__init__()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
            None,
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"
        self.rnn_type = rnn_type
        if rnn_type is not None:
            self.rnn = getattr(nn, rnn_type)(
                hidden_dim,
                rnn_dim,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            hdim = 2 * rnn_dim if bidirectional else rnn_dim
            self.linear_rnn = nn.Linear(
                hdim, hidden_dim
            )
            self.norm_rnn = choose_norm(norm_type, hidden_dim)
        
        self.attn = MultiHeadSelfAttention(
            hidden_dim,
            attention_dim=attention_dim,
            n_heads=att_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm_type, hidden_dim)

        self.feed_forward = PositionalwiseFeedForward(
            d_ffn=hidden_dim*expansion_factor,
            input_size=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.norm_ff = choose_norm(norm_type, hidden_dim)
        
        self.norm_output = choose_norm(norm_type, hidden_dim)

    def forward(self, x, flatten=False):
        """
        LSTM + Residual -> MultiheadAttention + Residual-> FFN + Residual -> Norm
        
        Args:
            x: Tensor of shape (batch, m, t, c)
            attn_mask: optional mask of shape (t, t)
        Returns:
            Tensor of shape (batch, m, t, c)
        """
        B, M, T, C = x.shape
        # ---- 统一 flatten ----
        # .view 在内存上不拷贝，只是改了shape
        x = rearrange(x, "b m t c -> (b m) t c")       # -> (B*M, T, C)
        
        # ---- RNN + Residual ----
        if self.rnn_type is not None:
            residual = x                                     # 共享内存，不拷贝
            x = self.norm_rnn(x)
            x, _ = self.rnn(x)                               # -> (B*M, T, H)
            x = self.linear_rnn(x)                           # -> (B*M, T, C)
            x = self.dropout(x)
            x = x + residual                                

        # ---- Attention + Residual ----
        residual = x
        x = self.norm_attn(x)
        x = self.attn(x, flatten=flatten)  
        x = self.dropout(x)
        x = x + residual

        # ---- FFN + Residual ----
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)                         # -> (B*M, T, C)
        x = self.dropout(x)
        x = x + residual

        # ---- reshape 回去 + 最后 Norm ----
        x = rearrange(x, "(b m) t c -> b m t c", b=B, m=M) # -> (B, M, T, C)
        return self.norm_output(x)
        
class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ---------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``hidden_dim``.
    hidden_dim : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, hidden_dim=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        output_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
        if output_size is None:
            output_size = input_size
        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, output_size),
        )

    def forward(self, x):
        """Applies PositionalwiseFeedForward to the input tensor x."""
        x = self.ffn(x)
        return x       

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, flatten=False):
        # get query, key, and value
        query, key, value = self.get_qkv(input)
        
        # rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)
        
        if flatten:
            q_shape = query.shape  # (B, T, H, D)
            # flatten 前两个维度（B*T, H, D）防止报错
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = value.flatten(0, 1)
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        try:
            with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except Exception as e:
            print("scaled_dot_product_attention failed with exception:")
            print(e)
            print("--- Debug info ---")
            print("query shape:", query.shape, "dtype:", query.dtype, "device:", query.device)
            print("key shape:", key.shape, "dtype:", key.dtype, "device:", key.device)
            print("value shape:", value.shape, "dtype:", value.dtype, "device:", value.device)
            print("Expected dtype: float16 or bfloat16, head_dim % 8 == 0")
            print("flash_attention_config:", self.flash_attention_config)
            raise
        if flatten:
            output = output.view(q_shape)  # (B, T, H, D)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key