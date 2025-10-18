from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch_complex.tensor import ComplexTensor
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcn import choose_norm
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.layers.septda import SepTDABlock, PositionalwiseFeedForward
from espnet2.enh.layers.rope_tda import RopeAttractorDecode
from espnet2.enh.layers.tda import AttractorDecode

attractors = {
    "tda": AttractorDecode,
    "ropetda": RopeAttractorDecode,
}

class SepformerTDAExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        # general setup
        input_dim: int= 256, # De=256
        hidden_dim: int = 128, # D=128
        output_dim: int = None, # De=256
        num_spk: int = 5,
        activation: str = "gelu",
        norm_type: str = "gLN",
        dual_layers: int = 1, # a dual-path processing block
        triple_layers: int = 8, # triple-path blocks N is 8
        segment_size: int = 96, # chunks of length K = 96
        dropout: float = 0.0,
        # rnn setup
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        rnn_dim: int = 256, # the number of hidden units in BLSTM is set to be 256 in each direction
        # self-attention setup
        att_heads: int = 4, # 4 attention heads
        attention_dim: int = 128, # the attention dimension is set to be 128
        flash_attention=False,
        # ffn setup
        expansion_factor: int = 4, # an expansion factor of 4 for the feed-forward module.
        # tda setup
        film_skip_connection: bool = False,
        attractor_type="tda", # "eda" or "tda"
        # multi-decoding setup
        multi_decode: bool = False,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        """SepTDA Separator

        Args:

        """
        super().__init__()

        self._num_spk = num_spk
        self.segment_size = segment_size
        self.linear = torch.nn.Linear(input_dim, hidden_dim) # input_dim -> hidden_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.segment_size = segment_size
        
        # rope
        assert attention_dim % att_heads == 0, (attention_dim, att_heads)
        rope_intra_chunk = RotaryEmbedding(attention_dim // att_heads)
        rope_inter_chunk = RotaryEmbedding(attention_dim // att_heads)
        # dual-path processing blocks
        self.dual_path = nn.ModuleList()
        for i in range(dual_layers):
            self.dual_path.append(
                SepTDABlock(
                    rope_intra_chunk,
                    rope_inter_chunk,
                    block_type="dual",
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    rnn_type=rnn_type,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,    
                    
                    att_heads=att_heads,
                    attention_dim=attention_dim,
                    flash_attention=flash_attention,
                    
                    expansion_factor=expansion_factor,
                )
            )
        # triple-path processing blocks
        self.triple_path = nn.ModuleList()
        for i in range(triple_layers):
            self.triple_path.append(
                SepTDABlock(
                    rope_intra_chunk,
                    rope_inter_chunk,
                    block_type="triple",
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    rnn_type=rnn_type,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,    
                    
                    att_heads=att_heads,
                    attention_dim=attention_dim,
                    flash_attention=flash_attention,
                    
                    expansion_factor=expansion_factor,
                )
            )
        
        # output layer
        self.norm_output = choose_norm(norm_type, hidden_dim)
        self.ffn_output = PositionalwiseFeedForward(
            d_ffn=hidden_dim*expansion_factor,
            input_size=hidden_dim,
            output_size=self.output_dim,
            dropout=dropout,
            activation=activation,
        )
        self.multi_decode = multi_decode
        if self.multi_decode:
            self.aux_norm_output = nn.ModuleList()
            self.aux_ffn_output = nn.ModuleList()
            self.aux_decoder = nn.ModuleList()
            for i in range(triple_layers - 1):
                self.aux_norm_output.append(choose_norm(norm_type, hidden_dim))
                self.aux_ffn_output.append(
                    PositionalwiseFeedForward(
                        d_ffn=hidden_dim*expansion_factor,
                        input_size=hidden_dim,
                        output_size=self.output_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                )
                self.aux_decoder.append(
                    ConvDecoder(
                        channel=self.output_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )

        # tda related params
        if attractor_type == "tda" or attractor_type == "ropetda":
            self.tda = attractors[attractor_type](hidden_dim, att_heads, 384, dropout, activation, chunk_shuffle=False)
        self.film = FiLM(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                skip_connection=film_skip_connection,
            )
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
        num_spk: int = None,
        task: str = None,
        speech_lengths: torch.Tensor = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, D_enc]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, num_spk, T, D_enc), ...]
            ilens (torch.Tensor): (B,)
        """

        feature = input # B, T, D_enc
        
        feature = self.linear(feature)  # B, T, D_enc -> B, T, D

        B, T, D = feature.shape

        feature = feature.transpose(1, 2)  # B, D, T
        segmented = self.split_feature(feature)  # chunking: B, D, K, S
        
        output = segmented
        batch, hidden_dim, K, S = output.shape 
        output = rearrange(output, "b d k s -> b k s d") # (B, K, S, D)
        
        # separation
        # dual-path processing
        for i in range(len(self.dual_path)):
            output = self.dual_path[i](output) # (B, K, S, D)
        # tda and film
        overlap_output = self.merge_feature(rearrange(output, "b k s d -> b d k s"), length=T) # (b d k s) -> (B, D, T)
        overlap_output = rearrange(overlap_output, "b d t -> b t d") # (B, T, D)
        attractors, probabilities = self.tda(overlap_output, num_spk)
        del overlap_output # overlap_output delete to reduce memory
        
        output = self.film(output, attractors[...,:-1,:]) # [B, C, K, S, D]
        # triple-path processing
        aux_batch = []
        for i in range(len(self.triple_path)):
            output = self.triple_path[i](output) # (B, C, K, S, D)
            # only for training
            if self.multi_decode and i < len(self.triple_path) - 1:
                aux_output = self.aux_ffn_output[i](self.aux_norm_output[i](output))
                aux_output = rearrange(
                    aux_output, "b c k s d -> (b c) d k s"
                )
                aux_output = self.merge_feature(aux_output, length=T)  # B*num_spk, D_enc, T overlap-add
                aux_output = rearrange(aux_output, "(b c) d t -> c b t d", b=B) # B*num_spk, D_enc, T -> num_spk, B, T, D_enc
                aux_batch.append(
                    [
                        self.aux_decoder[i](ps.to(torch.float32), speech_lengths)[0] for ps in aux_output # num_spk, B, T, D_enc -> num_spk, B, T
                    ]                 
                )
        
        output = self.ffn_output(self.norm_output(output)) # (B, C, K, S, D_enc)
        output = rearrange(
            output, "b c k s d -> (b c) d k s"
        )
        # B*num_spk, D_enc, K, S
        output = self.merge_feature(output, length=T)  # B*num_spk, D_enc, T overlap-add
        output = rearrange(output, "(b c) d t -> c b t d", b=B) # B*num_spk, D_enc, T -> num_spk, B, T, D_enc

        others = OrderedDict()
        if probabilities is not None:
            others["existance_probability"] = probabilities
        if self.multi_decode:
            others["aux_speech_pre"] = aux_batch
            
        return output, ilens, others

    def split_feature(self, x):
        B, D, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, D, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        B, D, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, D * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, D, length)

    @property
    def num_spk(self):
        return self._num_spk


class FiLM(nn.Module):
    def __init__(
        self,
        indim,
        enrolldim,
        filmdim,
        skip_connection=False,
    ):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(indim, filmdim),
            nn.PReLU(),
        )
        self.film_gamma = nn.Linear(enrolldim, filmdim)
        self.film_beta = nn.Linear(enrolldim, filmdim)
        self.linear2 = nn.Linear(filmdim, indim)
        self.skip_connection = skip_connection

    def forward(self, input, enroll_emb):
        # input: [B,T,F,D]
        # enroll_emb: [B,C,D]

        # FiLM params
        gamma = self.film_gamma(enroll_emb)  # [B,C,D]
        beta = self.film_beta(enroll_emb)    # [B,C,D]

        output = self.linear1(input)  # [B,T,F,D]

        # Rearrange for FiLM broadcasting
        output = rearrange(output, 'b t f d -> b 1 t f d')       # [B,1,T,F,D]
        gamma = rearrange(gamma, 'b c d -> b c 1 1 d')           # [B,C,1,1,D]
        beta = rearrange(beta, 'b c d -> b c 1 1 d')             # [B,C,1,1,D]

        # Apply FiLM
        output = gamma * output + beta                           # [B,C,T,F,D]

        output = self.linear2(output)                            # [B,C,T,F,D]

        # Optional skip connection
        if self.skip_connection:
            residual = rearrange(input, 'b t f d -> b 1 t f d')  # [B,1,T,F,D]
            output = output + residual                           # [B,C,T,F,D]

        return output