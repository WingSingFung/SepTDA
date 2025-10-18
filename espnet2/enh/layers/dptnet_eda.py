# The implementation of DPTNet-based MUSE model proposed in
# ***, in Proc. IEEE ASRU 2023.

import torch
import torch.nn as nn
import statistics

from espnet2.enh.layers.dptnet import ImprovedTransformerLayer
from espnet2.enh.layers.adapt_layers import make_adapt_layer


class DPTNet_EDA_Informed(nn.Module):
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
        i_eda_layer=4,
        num_eda_modules=1,
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

        # eda related params
        self.i_eda_layer = i_eda_layer
        self.num_eda_modules = num_eda_modules
        assert (
            self.num_eda_modules % 2 == 1
        ), "number of EDA modules should be odd number"
        if i_eda_layer is not None:
            self.sequence_aggregation = SequenceAggregation(input_size)
            self.eda = EncoderDecoderAttractor(input_size)

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
        # input shape: batch, N, dim1, dim2
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
            output = self.intra_chunk_process(output, i)
            output = self.inter_chunk_process(output, i)

            # compute attractor
            if i == self.i_eda_layer:
                aggregated_sequence = self.sequence_aggregation(
                    output.transpose(-1, -3)
                )
                attractors, probabilities = self.eda(
                    aggregated_sequence, num_spk=num_spk
                )
                output = (
                    output[..., None, :, :, :]
                    * attractors[..., :-1, :, None, None]
                )  # [B, J, N, L, K]
                # output, probabilities = self.eda_process(output, num_spk)
                output = output.view(-1, hidden_dim, dim1, dim2) # [B*J, N, L, K]
                batch = output.shape[0]

            # target speaker extraction part
            if i == self.i_adapt_layer and is_tse:
                # # assert num_spk > 1 or num_spk is not None, f"num_spk = {num_spk}"
                enroll_emb = enroll_emb.permute(0, 2, 3, 1)
                output = output.view(
                    org_batch, -1, hidden_dim, dim1, dim2
                ).permute(0, 1, 3, 4, 2)
                output = self.adapt_layer(output, enroll_emb)
                output = output.permute(0, 3, 1, 2)
                if self.adapt_layer_type == "attn_improved":
                    output = self.conditioning_model(
                        output, enroll_emb.mean(dim=(1, 2), keepdim=True)
                    )

        if self.i_eda_layer is None:
            probabilities = None
        output = self.output(output)  # [B*J, output_size, dim1, dim2]
        return output, probabilities

    def intra_chunk_process(self, x, layer_index):
        batch, N, chunk_size, n_chunks = x.size()
        x = (
            x.transpose(1, -1)
            .contiguous()
            .view(batch * n_chunks, chunk_size, N)
        )
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
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
        for i in range(self.num_eda_modules):
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
        for i in range(self.num_eda_modules):
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


class EncoderDecoderAttractor(nn.Module):
    def __init__(
        self,
        hidden_size,
        chunk_shuffle=True, # Only shuffle when the T is chunk_num
    ):
        super(EncoderDecoderAttractor, self).__init__()
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            batch_first=True,
        )

        self.attractor_existence_estimator = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        self.chunk_shuffle = chunk_shuffle

    def forward(self, input, num_spk=None):
        batch_size, C, H = input.shape
        output = input
        # if self.lstm_encoder.training:
        if self.chunk_shuffle:
            output = output[..., torch.randperm(C), :]  # shuffle chunk order
        _, state = self.lstm_encoder(output) # [1, batch_size, H]
        zero_input = input.new_zeros((batch_size, 1, H))
        outputs, existence_probabilities = [], []
        # estimate the number of speakers (inference)
        if num_spk is None:
            assert batch_size == 1, "We don't support batched computation in inference"
            existence_probability = 1
            while existence_probability > 0.5: # lstm 的缺点，需要多次迭代
                output, state = self.lstm_decoder(zero_input, state) # [batch_size, 1, H], [1, batch_size, H]
                existence_probability = self.attractor_existence_estimator(output) # [batch_size, 1, 1]
                existence_probabilities.append(existence_probability[..., 0]) # [batch_size]
                outputs.append(output[..., 0, :]) # [batch_size, H]
        # number of speakers is given (training)
        else:
            for j in range(num_spk + 1):
                output, state = self.lstm_decoder(zero_input, state)  # (batch, 1, hidden)
                output = output[..., 0, :]
                outputs.append(output) # [batch_size, H]
                existence_probability = self.attractor_existence_estimator(output)
                existence_probabilities.append(existence_probability[..., 0])
        outputs = torch.stack(outputs, dim=1)  # [B, J, H]
        existence_probabilities = torch.stack(existence_probabilities, dim=1)  # [B, J]
        return outputs, existence_probabilities # [batch_size, J, H], [batch_size, J]
