# The implementation of TDA

import torch
import torch.nn as nn
import statistics
import copy

from espnet.nets.pytorch_backend.nets_utils import get_activation

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoderAttractorLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0., activation="relu"):
        super(TransformerDecoderAttractorLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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

    def forward(self, attractor, src, src_key_padding_mask=None):
        # input:
        # src: [B, T, H]  attractor: [B, C, H]
        # Multi-head Self-attention
        C = attractor.size(1)  # C: number of speakers
        # 创建上三角为 True 的掩码，mask[i, j] 为 True 表示位置 i 不能看到位置 j
        mask = torch.triu(torch.ones(C, C, device=attractor.device, dtype=torch.bool), diagonal=1)
        attractor1 = self.self_attn(attractor, attractor, attractor, attn_mask=mask)[0]
        attractor = attractor + self.dropout3(attractor1)
        attractor = self.norm3(attractor)

        # Multi-head Source-target-attention
        attractor2 = self.attn(attractor, src, src, key_padding_mask=src_key_padding_mask)[0]
        attractor = attractor + self.dropout1(attractor2)
        attractor = self.norm1(attractor)

        # Feedforward
        attractor2 = self.linear2(self.dropout(self.activation(self.linear1(attractor))))
        attractor = attractor + self.dropout2(attractor2)
        attractor = self.norm2(attractor)
        return attractor

class TransformerDecoderAttractor(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, attractor, src, src_key_padding_mask=None): 
        output = attractor

        intermediate = []

        for layer in self.layers:
            output = layer(output, src, src_key_padding_mask)
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

class AttractorDecode(nn.Module):
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
        super(AttractorDecode, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        decoder_layer = TransformerDecoderAttractorLayer(
            hidden_size, n_heads, dim_feedforward, dropout, activation)
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
            while existence_probability > 0.5: # lstm 的缺点，需要多次迭代
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
            
            # attractor_output = torch.zeros((batch_size, num_spk + 1, H), device=input.device, dtype=input.dtype)
            # outputs = self.transformerdecoder(attractor_output, r_input) # shape: (batch_size, num_spk + 1, H)
            
            # existence_probabilities = self.attractor_existence_estimator(outputs)  # shape: (batch_size, num_spk + 1, 1)
            # existence_probabilities = existence_probabilities.squeeze(-1)         # shape: (batch_size, num_spk + 1)
        return outputs, existence_probabilities # [batch_size, J, H], [batch_size, J]
