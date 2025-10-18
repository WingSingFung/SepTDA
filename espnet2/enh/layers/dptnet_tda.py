# The implementation of DPTNet-based MUSE model proposed in
# ***, in Proc. IEEE ASRU 2023.

import torch
import torch.nn as nn
import statistics
import copy

from espnet2.enh.layers.dptnet import ImprovedTransformerLayer
from espnet2.enh.layers.adapt_layers import make_adapt_layer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet2.enh.layers.rope_tda import RopeAttractorDecode
from espnet2.enh.layers.tda import AttractorDecode

attractors = {
    "tda": AttractorDecode,
    "ropetda": RopeAttractorDecode,
}

class DPTNet_TDA_Informed(nn.Module):
    """Dual-path transformer network.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_layers=1,
        bidirectional=True,
        norm_type="gLN",
        i_tda_layer=4,
        num_tda_modules=1,
        attractor:str="tda", # tda or ropetda
        i_adapt_layer=4,
        adapt_layer_type="attn",
        adapt_enroll_dim=64,
        adapt_attention_dim=512,
        adapt_hidden_dim=512,
        adapt_softmax_temp=1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.output_size = output_size
        self.output_size = input_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.chan_transformer = nn.ModuleList()
        for i in range(num_layers):
            self.row_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=True,
                    norm=norm_type,
                )
            )  # intra-segment RNN is always noncausal
            self.col_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )

        # output layer
        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size, 1)
        )

        # tda related params
        self.i_tda_layer = i_tda_layer
        self.num_tda_modules = num_tda_modules
        assert (
            self.num_tda_modules % 2 == 1
        ), "number of tda modules should be odd number"
        if i_tda_layer is not None:
            self.sequence_aggregation = SequenceAggregation(input_size)
            self.tda = attractors[attractor](
                input_size,
                att_heads,
                hidden_size,
                dropout,
                activation,
            )

        # tse related params
        self.i_adapt_layer = i_adapt_layer
        if i_adapt_layer is not None:
            assert adapt_layer_type in ["attn", "attn_improved"]
            self.adapt_enroll_dim = adapt_enroll_dim
            self.adapt_layer_type = adapt_layer_type
            # set parameters
            adapt_layer_params = {
                "attention_dim": adapt_attention_dim,
                "hidden_dim": adapt_hidden_dim,
                "softmax_temp": adapt_softmax_temp,
                "is_dualpath_process": True,
            }
            # prepare additional processing block
            if adapt_layer_type == "attn_improved":
                self.conditioning_model = ConditionalDPTNet(
                    rnn_type=rnn_type,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    att_heads=att_heads,
                    dropout=dropout,
                    activation=activation,
                    num_layers=2,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                    enroll_size=input_size,
                    conditioning_size=512,
                )
                adapt_layer_type = "attn"
            # load speaker selection module
            self.adapt_layer = make_adapt_layer(
                adapt_layer_type,
                indim=input_size,
                enrolldim=adapt_enroll_dim,
                ninputs=1,
                adapt_layer_kwargs=adapt_layer_params,
            )

    def forward(self, input, enroll_emb, num_spk=None):
        # input shape: batch, N, dim1, dim2,  N is the number of chunk, each chunk has K length and dim1=K, dim2=Hidden size
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)

        # task flag
        is_tse = enroll_emb is not None
        # processing
        output = input
        batch, hidden_dim, dim1, dim2 = output.shape 
        org_batch = batch
        for i in range(len(self.row_transformer)):
            output = self.intra_chunk_process(output, i) # [b, N, chunk_size, n_chunks]
            output = self.inter_chunk_process(output, i) # [b, N, chunk_size, n_chunks]

            # compute attractor
            if i == self.i_tda_layer:
                aggregated_sequence = self.sequence_aggregation(
                    output.transpose(-1, -3) # [b, N, chunk_size, n_chunks] -> [b, n_chunks, chunk_size, N]
                ) # 降维到3维 (B, n_chunks, N)
                attractors, probabilities = self.tda(
                    aggregated_sequence, num_spk=num_spk
                )
                output = (
                    output[..., None, :, :, :]
                    * attractors[..., :-1, :, None, None]
                )  # [B, J, N, L, K] 
                # output, probabilities = self.eda_process(output, num_spk)
                output = output.view(-1, hidden_dim, dim1, dim2) # [b * J, N, L, K]
                batch = output.shape[0]

        if self.i_tda_layer is None:
            probabilities = None
        output = self.output(output)  # B, output_size, dim1, dim2
        return output, probabilities

    def intra_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = (
            x.transpose(1, -1)
            .contiguous()
            .view(batch * n_chunks, chunk_size, N)
        )
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1) # [b, N, chunk_size, n_chunks]
        return x

    def inter_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch * chunk_size, n_chunks, N)
        )
        x = self.col_transformer[layer_index](x)
        x = x.view(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return x

    def eda_process(self, x, num_spk):
        num_attractors = []
        attractors = []
        probabilities = []
        for i in range(self.num_tda_modules):
            aggregated_sequence = self.sequence_aggregation[i](
                x.transpose(-1, -3)
            )
            attractor, probability = self.eda[i](
                aggregated_sequence, num_spk=num_spk
            )
            attractors.append(attractor)
            probabilities.append(probability)
            num_attractors.append(
                attractor.shape[-2]
            )  # estimated number of speakers
        # we use mode value as the estimated number of speakers
        output, count = 0.0, 0
        est_num_spk = statistics.mode(num_attractors)
        for i in range(self.num_tda_modules):
            if num_attractors[i] == est_num_spk:
                output = output + (
                    x[..., None, :, :, :]
                    * attractors[i][..., :-1, :, None, None]
                )  # [B, J, N, L, K]
                count += 1
        output = output / count
        probabilities = torch.cat(
            probabilities, dim=0
        )  # concat along batch dim
        return output, probabilities


class ConditionalDPTNet(nn.Module):
    """Dual-path transformer network.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_layers=1,
        bidirectional=True,
        norm_type="gLN",
        # film related
        enroll_size=None,
        conditioning_size=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if enroll_size is None:
            enroll_size = input_size
        if conditioning_size is None:
            conditioning_size = input_size
        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.film = nn.ModuleList()
        for i in range(num_layers):
            self.row_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=True,
                    norm=norm_type,
                )
            )  # intra-segment RNN is always noncausal
            self.col_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )
            self.film.append(
                FiLM(
                    input_size,
                    enroll_size,
                    conditioning_size,
                )
            )
        # output layer
        # self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input, enroll_emb):
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        output = input
        for i in range(len(self.row_transformer)):
            output = self.film[i](
                output.transpose(-1, -3), enroll_emb
            ).transpose(-1, -3)
            output = self.intra_chunk_process(output, i)
            output = self.inter_chunk_process(output, i)

        # output = self.output(output)  # B, output_size, dim1, dim2
        return output

    def intra_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
        return x

    def inter_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        x = self.col_transformer[layer_index](x)
        x = x.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return x


class FiLM(nn.Module):
    def __init__(
        self,
        indim,
        enrolldim,
        filmdim,
        skip_connection=True,
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
        # FiLM params
        gamma = self.film_gamma(enroll_emb)
        beta = self.film_beta(enroll_emb)
        # input processing
        output = input
        output = self.linear1(output)
        output = output * gamma + beta
        output = self.linear2(output)
        if self.skip_connection:
            output = output + input
        return output


class SequenceAggregation(nn.Module):
    def __init__(
        self,
        hidden_size,
        r=4,
    ):
        super(SequenceAggregation, self).__init__()
        self.path1 = nn.Sequential(
            nn.Linear(hidden_size, r * hidden_size),
            nn.Tanh(),
            nn.Linear(r * hidden_size, r),
            nn.Softmax(dim=-1),
        )
        self.linear = nn.Linear(hidden_size, hidden_size // r)

    def forward(self, input):
        batch_size, num_segments, segment_size, hidden_size = input.shape
        alpha = self.path1(input)
        W = torch.matmul(alpha.transpose(-1, -2), self.linear(input)).reshape(batch_size, num_segments, hidden_size)
        return W

