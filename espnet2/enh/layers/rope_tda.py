# The implementation of Rope TDA

import torch
import torch.nn as nn
import statistics
import copy
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from rotary_embedding_torch import RotaryEmbedding

from espnet.nets.pytorch_backend.nets_utils import get_activation

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadAttention(nn.Module):
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

        # query 只从 input 投影
        self.q_proj = nn.Linear(emb_dim, attention_dim, bias=False)
        # key/value 从 key_value_input 投影
        self.kv_proj = nn.Linear(emb_dim, attention_dim * 2, bias=False)

        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False),
            nn.Dropout(dropout)
        )

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, key_value_input=None, attn_mask: torch.Tensor = None, flatten=False):
        """
        input: Tensor of shape (B, T_q, D)
        key_value_input: Tensor of shape (B, T_kv, D), or None for self-attention
        """
        # 在 get_qkv 里处理 key_value_input 是否为 None
        query, key, value = self.get_qkv(input, key_value_input)

        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        if flatten:
            q_shape = query.shape  # (B*T, H, D) after flatten
            query = query.flatten(0, 1)
            key   = key.flatten(0, 1)
            value = value.flatten(0, 1)

        try:
            with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                attn_output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except Exception as e:
            print("scaled_dot_product_attention error:", e)
            raise

        if flatten:
            attn_output = attn_output.view(q_shape)

        # attn_output: (B, H, T_q, D_h) → (B, T_q, H*D_h)
        out = attn_output.transpose(1, 2).reshape(attn_output.size(0), attn_output.size(2), -1)
        return self.aggregate_heads(out)

    def get_qkv(self, input, key_value_input=None):
        """
        统一在这里分出 query, key, value：
        - key_value_input=None 时视为 self-attention
        - 否则为 cross-attention
        返回：
          query: (B, H, T_q, D_h)
          key:   (B, H, T_kv, D_h)
          value: (B, H, T_kv, D_h)
        """
        if key_value_input is None:
            key_value_input = input

        B, T_q, _ = input.shape
        T_kv = key_value_input.size(1)
        D_h = self.q_proj.out_features // self.n_heads

        # query: (B, T_q, H, D_h) → (B, H, T_q, D_h)
        query = self.q_proj(input) \
                    .reshape(B, T_q, self.n_heads, D_h) \
                    .transpose(1, 2)

        # kv: (B, T_kv, 2, H, D_h) → permute → (2, B, H, T_kv, D_h)
        kv = self.kv_proj(key_value_input) \
                 .reshape(B, T_kv, 2, self.n_heads, D_h) \
                 .permute(2, 0, 3, 1, 4)

        key, value = kv[0], kv[1]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key   = self.rope.rotate_queries_or_keys(key)
        return query, key

class TransformerDecoderAttractorLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0., activation="relu", rope=None):
        super(TransformerDecoderAttractorLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_model, nhead, dropout=dropout, rope=rope)
        self.attn = MultiHeadAttention(d_model, d_model, nhead, dropout=dropout, rope=rope)

        # Implementation of Feedforward Layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation.lower() == "linear":
            self.activation = nn.Identity()
        else:
            self.activation = get_activation(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderAttractorLayer, self).__setstate__(state)

    def forward(self, attractor, src):
        # input:
        # src: [B, T, H]  attractor: [B, C, H]
        # Multi-head Self-attention
        C = attractor.size(1)  # C: number of speakers
        # 创建上三角为 True 的掩码，mask[i, j] 为 True 表示位置 i 不能看到位置 j
        mask = torch.triu(torch.ones(C, C, device=attractor.device, dtype=torch.bool), diagonal=1)
        attractor1 = self.self_attn(attractor, attn_mask=mask)
        attractor = attractor + self.dropout3(attractor1)
        attractor = self.norm3(attractor)

        # Multi-head Source-target-attention
        attractor1 = self.attn(attractor, key_value_input=src)
        attractor = attractor + self.dropout1(attractor1)
        attractor = self.norm1(attractor)

        # Feedforward
        attractor1 = self.linear2(self.dropout(self.activation(self.linear1(attractor))))
        attractor = attractor + self.dropout2(attractor1)
        attractor = self.norm2(attractor)
        return attractor

class TransformerDecoderAttractor(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, attractor, src): 
        output = attractor

        intermediate = []

        for layer in self.layers:
            output = layer(output, src)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class RopeAttractorDecode(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads=4,
        dim_feedforward=2048,
        dropout=0.5,
        activation="relu",
        depth=2,
        chunk_shuffle=True, # Only shuffle when the T is chunk_num
    ):
        super(RopeAttractorDecode, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        rope = RotaryEmbedding(hidden_size // n_heads)
        decoder_layer = TransformerDecoderAttractorLayer(
            hidden_size, n_heads, dim_feedforward, dropout, activation, rope)
        self.transformerdecoder = TransformerDecoderAttractor(decoder_layer, depth)
        self.attractor_existence_estimator = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        self.chunk_shuffle = chunk_shuffle

    def forward(self, input, num_spk=None):
        batch_size, C, H = input.shape
        r_input = input
        if self.chunk_shuffle:
            r_input = r_input[..., torch.randperm(C), :]  # shuffle chunk order
        
        outputs, existence_probabilities = [], []
        # estimate the number of speakers (inference)
        if num_spk is None:
            attractor_output = torch.zeros((batch_size, 1, H), device=input.device, dtype=input.dtype)
            assert batch_size == 1, "We don't support batched computation in inference"
            existence_probability = 1
            while existence_probability > 0.5:
                output = self.transformerdecoder(attractor_output, r_input) # 
                existence_probability = self.attractor_existence_estimator(output[..., -1, :]) # [batch_size, 1]
                existence_probabilities.append(existence_probability[..., 0]) # [batch_size]
                outputs.append(output[..., -1, :]) # [batch_size, H]
                attractor_output = torch.cat([attractor_output, output[..., -1, :].unsqueeze(1)], dim=1) # [batch_size, j, H]
            outputs = torch.stack(outputs, dim=1)  # [B, J, H]
            existence_probabilities = torch.stack(existence_probabilities, dim=1)  # [B, J]
        # number of speakers is given (training)
        else:
            attractor_output = torch.zeros((batch_size, 1, H), device=input.device, dtype=input.dtype)
            for j in range(num_spk+1):
                output = self.transformerdecoder(attractor_output, r_input)
                existence_probability = self.attractor_existence_estimator(output[..., -1, :])
                existence_probabilities.append(existence_probability[..., 0])
                outputs.append(output[..., -1, :])
                attractor_output = torch.cat([attractor_output, output[..., -1, :].unsqueeze(1)], dim=1)
            outputs = torch.stack(outputs, dim=1)
            existence_probabilities = torch.stack(existence_probabilities, dim=1)

        return outputs, existence_probabilities # [batch_size, J, H], [batch_size, J]