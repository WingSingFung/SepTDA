# noqa E501: Ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/src/models/adapt_layers.py
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from functools import partial

import torch
import torch.nn as nn


def make_adapt_layer(type, indim, enrolldim, ninputs=1, adapt_layer_kwargs={}):
    adapt_class = adaptation_layer_types.get(type)
    return adapt_class(indim, enrolldim, ninputs, **adapt_layer_kwargs)


def into_tuple(x):
    """Transforms tensor/list/tuple into tuple."""
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("x should be tensor, list of tuple")


def into_orig_type(x, orig_type):
    """Inverts into_tuple function."""
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class ConcatAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1):
        super().__init__()
        self.ninputs = ninputs
        self.transform = nn.ModuleList(
            [nn.Linear(indim + enrolldim, indim) for _ in range(ninputs)]
        )

    def forward(self, main, enroll):
        """ConcatAdaptLayer forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for transform, main0, enroll0 in zip(self.transform, main, enroll):
            out.append(
                transform(
                    torch.cat(
                        (main0, enroll0[:, :, None].expand(main0.shape)), dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
            )
        return into_orig_type(tuple(out), orig_type)


class MulAddAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1, do_addition=True):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        if do_addition:
            assert enrolldim == 2 * indim, (enrolldim, indim)
        else:
            assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll):
        """MulAddAdaptLayer Forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs, (
            len(main),
            len(enroll),
            self.ninputs,
        )
        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(
                    enroll0_mul[:, :, None] * main0 + enroll0_add[:, :, None]
                )
            else:
                out.append(enroll0[:, :, None] * main0)
        return into_orig_type(tuple(out), orig_type)


class AttentionAdaptLayer(nn.Module):
    def __init__(
        self,
        indim,
        enrolldim,
        ninputs=1,
        softmax_temp=1,
        attention_dim=200,
        hidden_dim=200,
        is_dualpath_process=False,
        return_attn=False,
    ):
        """
        AttentionAdaptLayer for speaker selection in target speaker extraction.
        https://ieeexplore.ieee.org/abstract/document/8683448

        Args:
            indim: int,
                Input hidden dimension.
            enrolldim: int
                Hidden dimension of enrollment embedding.
            ninputs: int, optional
                The number of inputs (default: ``1``).
            softmax_temp: int, optional
                Temprature of softmax funcion (default: ``1``).
            attention_dim: int, optional
                Hidden dimension of attention (default: ``200``).
            hidden_dim: int, optional
                Hidden dimension in MLP layers (default: ``200``).
            is_dualpath_process: bool, optonal
                Whether the backbone model is dual-path model or not (default: ``False``).
            return_attn: bool, optional
                If ``True``, attention weight is also returned (default:``False``).
        """
        super().__init__()
        self.return_attn = return_attn
        self.mlp_v = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )
        self.mlp_iv = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )
        self.mlp_aux = nn.Sequential(
            nn.Linear(enrolldim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )

        self.W_v = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_iv = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_aux = nn.Linear(attention_dim, attention_dim, bias=False)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.b = nn.Parameter(torch.randn(attention_dim))

        self.attention_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = softmax_temp
        self.indim = indim
        self.is_dualpath_process = is_dualpath_process

        # assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll=None):
        """AttentionAdaptLayer Forward. Variable names follow the paper.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        outputs = []
        orig_type = type(main)
        is_tse = enroll is not None

        if not isinstance(main, tuple):
            main = (main,)
            enroll = (enroll,)
        for main0, enroll0 in zip(main, enroll):
            # non-dualpath case:
            #   main: (..., nspk, time, hidden)
            #   enroll: (..., time, hidden)
            # dual-path case:
            #   main: (..., nspk, chunk_size, num_chunk, hidden)
            #   enroll: (..., chunk_size, num_chunk, hidden)
            if is_tse:
                assert type(main0) == type(enroll0)
                if self.is_dualpath_process:
                    (
                        batch,
                        num_spk,
                        chunk_size,
                        num_chunks,
                        hidden,
                    ) = main0.shape
                    main0 = main0.permute(0, 2, 3, 1, 4)
                    mean_dim = (1, 2)
                else:
                    batch, num_spk, time, hidden = main0.shape
                    main0 = main0.transpose(-2, -3)
                    mean_dim = (1,)
                s_v = self.mlp_v(main0)  # (..., time, nspk, hidden)
                s_iv = self.mlp_iv(main0).mean(
                    dim=mean_dim, keepdim=True
                )  # (..., 1, nspk, hidden)
                s_aux = self.mlp_aux(enroll0).mean(dim=mean_dim, keepdim=True)

                # e: [batch, chunk_size, num_chunk, n_enroll, nspk, hidden]
                # a: [batch, chunk_size, num_chunk, n_enroll, nspk, 1]
                e_v = self.W_v(
                    s_v
                )  # (batch, chunk_size, num_chunk, nspk, hidden)
                e_iv = self.W_iv(s_iv)  # (..., nspk, hidden)
                e_aux = self.W_aux(
                    s_aux
                )  # (batch, 1, 1, hidden) / (batch, 1, hidden)
                e = (
                    e_v[..., None, :, :]
                    + e_iv[..., None, :, :]
                    + e_aux[..., None, None, :, :]
                    + self.b
                )
                a = self.w(self.attention_activation(e))
                a = self.softmax(a * self.alpha)
                out = (a * main0[..., None, :, :]).sum(dim=-2)[..., 0, :]
            else:
                if self.is_dualpath_process:
                    (
                        batch,
                        chunk_size,
                        num_chunks,
                        num_spk,
                        hidden,
                    ) = main0.shape
                    zeros = main0.new_zeros(
                        (batch, chunk_size, num_chunks, 1, 1, 1)
                    )
                    ones = main0.new_ones(
                        (batch, chunk_size, num_chunks, 1, 1, 1)
                    )
                else:
                    batch, time, num_spk, hidden = main0.shape
                    zeros = main0.new_zeros((batch, time, 1, 1, 1))
                    ones = main0.new_ones((batch, time, 1, 1, 1))
                a = torch.cat(
                    (
                        torch.cat((ones, zeros), dim=-2),
                        torch.cat((zeros, ones), dim=-2),
                    ),
                    dim=-3,
                )
                out = (a * main0[..., None, :, :]).sum(dim=-2)
            outputs.append(out)
        if self.return_attn:
            return into_orig_type(tuple(outputs), orig_type), a
        else:
            return into_orig_type(tuple(outputs), orig_type)


# aliases for possible adaptation layer types
adaptation_layer_types = {
    "concat": ConcatAdaptLayer,
    "muladd": MulAddAdaptLayer,
    "mul": partial(MulAddAdaptLayer, do_addition=False),
    "attn": AttentionAdaptLayer,
}
