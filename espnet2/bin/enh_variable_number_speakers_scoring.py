#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
# from mir_eval.separation import bss_eval_sources
from fast_bss_eval import bss_eval_sources, si_bss_eval_sources
from pystoi import stoi
from pesq import pesq
from pesq import PesqError

from typeguard import check_argument_types

from espnet2.enh.loss.criterions.time_domain import SISNRLoss,SNRLoss
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.train.dataset import kaldi_loader
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args

si_snr_loss = SISNRLoss()
snr_loss = SNRLoss()
DUMMY_SYNBOL = "dummy"
use_tensor = True

def get_readers(scps: List[str], dtype: str):
    # Determine the audio format (sound or kaldi_ark)
    with open(scps[0], "r") as f:
        line = f.readline()
        filename = Path(line.strip().split(maxsplit=1)[1]).name
    if re.fullmatch(r".*\.ark(:\d+)?", filename):
        # xxx.ark or xxx.ark:123
        readers = [kaldi_loader(f, float_dtype=dtype) for f in scps]
        audio_format = "kaldi_ark"
    else:
        readers = [SoundScpReader(f, dtype=dtype) for f in scps]
        audio_format = "sound"
    return readers, audio_format


def read_audio(reader, key, audio_format="sound"):
    if audio_format == "sound":
        return reader[key][1]
    elif audio_format == "kaldi_ark":
        return reader[key]
    else:
        raise ValueError(f"Unknown audio format: {audio_format}")


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
    flexible_numspk: bool,
):
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if not flexible_numspk:
        assert len(ref_scp) == len(inf_scp), (ref_scp, inf_scp)
    num_spk = len(inf_scp)

    keys = [
        line.rstrip().split(maxsplit=1)[0] for line in open(key_file, encoding="utf-8")
    ]

    ref_readers, ref_audio_format = get_readers(ref_scp, dtype)
    inf_readers, inf_audio_format = get_readers(inf_scp, dtype)

    # get sample rate
    retval = ref_readers[0][keys[0]]
    if ref_audio_format == "kaldi_ark":
        sample_rate = ref_readers[0].rate
    elif ref_audio_format == "sound":
        sample_rate = retval[0]
    else:
        raise NotImplementedError(ref_audio_format)
    assert sample_rate is not None, (sample_rate, ref_audio_format)

    # check keys
    # if not flexible_numspk:
    #     for inf_reader, ref_reader in zip(inf_readers, ref_readers):
    #         assert inf_reader.keys() == ref_reader.keys()

    with DatadirWriter(output_dir) as writer:
        for n, key in enumerate(keys):
            logging.info(f"[{n}] Scoring {keys}")
            # for inf_reader in inf_readers:
            #     print(inf_reader.data[key], flush=True)
            if not flexible_numspk:
                ref_audios = [
                    read_audio(ref_reader, key, audio_format=ref_audio_format)
                    for ref_reader in ref_readers
                    if ref_reader.data[key] != DUMMY_SYNBOL
                ]
                inf_audios = [
                    read_audio(inf_reader, key, audio_format=inf_audio_format)
                    for (inf_reader, ref_reader) in zip(inf_readers, ref_readers)
                    if ref_reader.data[key] != DUMMY_SYNBOL
                ]
            else:
                ref_audios = [
                    read_audio(ref_reader, key, audio_format=ref_audio_format)
                    for ref_reader in ref_readers
                    if key in ref_reader.keys() and ref_reader.data[key] != DUMMY_SYNBOL
                ]
                inf_audios = [
                    read_audio(inf_reader, key, audio_format=inf_audio_format)
                    for inf_reader in inf_readers
                    if key in inf_reader.keys() and inf_reader.data[key] != DUMMY_SYNBOL
                ]
            ref = np.array(ref_audios)
            inf = np.array(inf_audios)
            # print(">>>>>", ref.shape, inf.shape, flush=True)
            if ref.ndim > inf.ndim:
                # multi-channel reference and single-channel output
                ref = ref[..., ref_channel]
            elif ref.ndim < inf.ndim:
                # single-channel reference and multi-channel output
                inf = inf[..., ref_channel]
            elif ref.ndim == inf.ndim == 3:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]
            # All of these have been done in the inference stage
            assert ref.shape == inf.shape, (ref.shape, inf.shape)
            # if not flexible_numspk:
            #     assert ref.shape == inf.shape, (ref.shape, inf.shape)
            # else:
            #     # epsilon value to avoid divergence
            #     # caused by zero-value, e.g., log(0)
            #     eps = 0.000001
            #     # if num_spk of ref > num_spk of inf
            #     if ref.shape[0] > inf.shape[0]: # underestimation
            #         # MUSE:
            #         # In contrast, when the model underestimated the number (i.e., Nˆ < N ),
            #         # we first computed the optimal assignment of estimate-reference pairs,
            #         # and then compensated for underestimated speeches by copying 
            #         # one of the Nˆ estimated signals that gave the highest SDR with the 
            #         # reference speech for each of the remaining N − Nˆ references.
            #         p = np.full((ref.shape[0] - inf.shape[0], inf.shape[1]), eps)
            #         inf = np.concatenate([inf, p])
            #         num_spk = ref.shape[0]
            #         # SepTDA:
            #         # In cases of an underestimation in the number of speakers 
            #         # (i.e., Cˆ < C), a silence signal (i.e., an all-zeros signal) 
            #         # is used as the estimate for the speakers that were not estimated 
            #         # as in [28]. Since the SI-SDR and SDR metrics are undefined 
            #         # if the reference signal is all-zero (i.e., log(0) is undefined), 
            #         # an epsilon value of 10−8 is used for the final log calculation, 
            #         # resulting in an SI-SDR value of −80 dB.
            #     # if num_spk of ref < num_spk of inf
            #     elif ref.shape[0] < inf.shape[0]: # overestimation
            #         # MUSE:
            #         # When the model overestimated the number (i.e., Nˆ > N ), 
            #         # we computed the optimal assignment of estimatereference pairs 
            #         # and ignored the rest Nˆ − N speakers.
            #         p = np.full((inf.shape[0] - ref.shape[0], ref.shape[1]), eps)
            #         ref = np.concatenate([ref, p])
            #         num_spk = inf.shape[0]
            #         # SepTDA:
            #         # When there is an overestimation of the number of speakers 
            #         # (i.e., Cˆ > C), we select the estimated sources corresponding 
            #         # to the initial C speaker queries.
            #     else:
            #         num_spk = ref.shape[0]
            
            # permutation is already solved in inference stage
            if use_tensor:
                tensor_ref = torch.from_numpy(ref).float()
                tensor_inf = torch.from_numpy(inf).float()
                # gpu is available
                if torch.cuda.is_available():
                    tensor_ref = tensor_ref.to("cuda:0")
                    tensor_inf = tensor_inf.to("cuda:0")
                sdr, sir, sar = bss_eval_sources(tensor_ref, tensor_inf, compute_permutation=False, clamp_db=80)
                si_snr, si_sir, si_sar = si_bss_eval_sources(tensor_ref, tensor_inf, compute_permutation=False, clamp_db=80)
                sdr, sir, sar = sdr.cpu().numpy(), sir.cpu().numpy(), sar.cpu().numpy()
                si_snr, si_sir, si_sar = si_snr.cpu().numpy(), si_sir.cpu().numpy(), si_sar.cpu().numpy()
            else:
                sdr, sir, sar = bss_eval_sources(ref, inf, compute_permutation=False, clamp_db=80)
                si_snr, si_sir, si_sar = si_bss_eval_sources(ref, inf, compute_permutation=False, clamp_db=80)
            
            perm = np.arange(num_spk)

            for i in range(num_spk):
                if i < ref.shape[0]:
                    stoi_score = stoi(ref[i], inf[i], fs_sig=sample_rate)
                    estoi_score = stoi(
                        ref[i], inf[i], fs_sig=sample_rate, extended=True
                    )
                    snr_score = -float(
                        snr_loss(
                            torch.from_numpy(ref[i][None,...]),
                            torch.from_numpy(inf[i][None,...]),
                        )) 
                    # si_snr_score = -float(
                    #     si_snr_loss(
                    #         torch.from_numpy(ref[i][None, ...]),
                    #         torch.from_numpy(inf[i][None, ...]),
                    #     )
                    # )
                    writer[f"STOI_spk{i + 1}"][key] = str(stoi_score * 100)  # in percentage
                    writer[f"ESTOI_spk{i + 1}"][key] = str(estoi_score * 100)
                    writer[f"SNR_spk{i + 1}"][key] = str(snr_score)
                    # writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr_score)
                    writer[f"SI_SNR_spk{i + 1}"][key] = str(si_snr[i])
                    writer[f"SI_SAR_spk{i + 1}"][key] = str(si_sar[i])
                    writer[f"SI_SIR_spk{i + 1}"][key] = str(si_sir[i])
                    writer[f"SDR_spk{i + 1}"][key] = str(sdr[i])
                    writer[f"SAR_spk{i + 1}"][key] = str(sar[i])
                    writer[f"SIR_spk{i + 1}"][key] = str(sir[i])
                    
                    # PESQ
                    if sample_rate == 16000:
                        wbpesq_score = pesq(sample_rate, ref[i], inf[i], mode="wb", on_error=PesqError.RETURN_VALUES,)
                    nbpesq_score = pesq(sample_rate, ref[i], inf[i], mode="nb", on_error=PesqError.RETURN_VALUES,)
                    if sample_rate == 16000:
                        if wbpesq_score == PesqError.NO_UTTERANCES_DETECTED:
                            print(key, flush=True)
                        else:
                            writer[f"WBPESQ_spk{i + 1}"][key] = str(wbpesq_score)  # in percentage
                    if nbpesq_score == PesqError.NO_UTTERANCES_DETECTED:
                        print(key, flush=True)
                    else:
                        writer[f"NBPESQ_spk{i + 1}"][key] = str(nbpesq_score)

                    # save permutation assigned script file
                    if i < len(ref_scp):
                        if inf_audio_format == "sound":
                            writer[f"wav_spk{i + 1}"][key] = inf_readers[perm[i]].data[key]
                        elif inf_audio_format == "kaldi_ark":
                            # NOTE: SegmentsExtractor is not supported
                            writer[f"wav_spk{i + 1}"][key] = inf_readers[
                                perm[i]
                            ].loader._dict[key]
                        else:
                            raise ValueError(f"Unknown audio format: {inf_audio_format}")
                else:
                    # write dummy scores
                    writer[f"STOI_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"ESTOI_spk{i + 1}"][key] = DUMMY_SYNBOL
                    if sample_rate == 16000:
                        writer[f"WBPESQ_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"NBPESQ_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SNR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SI_SNR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SI_SIR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SI_SAR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SDR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SAR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"SIR_spk{i + 1}"][key] = DUMMY_SYNBOL
                    writer[f"wav_spk{i + 1}"][key] = DUMMY_SYNBOL


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str)
    group.add_argument("--ref_channel", type=int, default=0)
    group.add_argument("--flexible_numspk", type=str2bool, default=False)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    scoring(**kwargs)


if __name__ == "__main__":
    main()
