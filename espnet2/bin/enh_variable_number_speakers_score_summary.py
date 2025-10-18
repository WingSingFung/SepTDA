#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import json

from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.read_text import read_2columns_text


def has_speaker_duplicates(lst):
    # no duplicates if lst has only one compoenent
    if len(lst) <= 1:
        return False

    # remove duplication by changing into set
    unique_set = set(lst)
    # if length is different, there was duplication
    return len(lst) != len(unique_set)


def filter_result(result, filter_type: str = None):
    assert filter_type in [None, "with_duplication", "without_duplication"]
    # return input as it is
    if filter_type is None:
        return result
    # filtering
    filtered_result = {}
    for key, score in result.items():
        # first check whether there is a speaker overlap
        nspk = int(key[0])
        spk_ids = key.split("_")[1 : (nspk + 1)]
        has_duplicate = has_speaker_duplicates(spk_ids)
        # then do filtering
        if (filter_type == "with_duplication" and has_duplicate) or (
            filter_type == "without_duplication" and not has_duplicate
        ):
            filtered_result[key] = score
    return filtered_result


def score_summary(args, filter_type: str = None):
    # get scores for all number of speakers
    # write results to some files
    results = {}
    for protocol in args.protocols:
        if protocol == "Est_num_spk":
            scores = read_2columns_text(args.score_dir / f"{protocol}")
            scores = filter_result(scores, filter_type=filter_type)
            results[protocol] = speaker_count(scores, args.max_num_spk)
        else:
            scores = []
            for spk in range(1, args.max_num_spk + 1):
                score = read_2columns_text(
                    args.score_dir / f"{protocol}_spk{spk}"
                )
                score = filter_result(score, filter_type=filter_type)
                scores.append(score)
            results[protocol] = get_score(scores, args.max_num_spk)
    return results


def get_score(scores, max_num_spk):
    # get average score for speech metrics
    score_total, count = {}, {}
    for score in scores:
        for key, value in score.items():
            # first word of key must be like 2mix, 3mix, ...
            nmix = int(key[0])
            assert nmix <= max_num_spk, (nmix, max_num_spk)
            if value == "dummy":
                continue
            if nmix not in score_total.keys():
                score_total[nmix] = float(value)
                count[nmix] = 1
            else:
                score_total[nmix] += float(value)
                count[nmix] += 1
    for nmix in score_total:
        score_total[nmix] = str(round(score_total[nmix] / count[nmix], 4))
    return score_total


def speaker_count(score, max_num_spk):
    # speaker counting accuracy
    # under, over estimation
    speaker_count = {}
    for key, est_num_spk in score.items():
        # first word of key must be like 2mix, 3mix, ...
        nmix = int(key[0])
        if nmix not in speaker_count.keys():
            speaker_count[nmix] = {}
        if est_num_spk not in speaker_count[nmix].keys():
            speaker_count[nmix][est_num_spk] = 1
        else:
            speaker_count[nmix][est_num_spk] += 1
    # compute percentage
    for nmix, counts in speaker_count.items():
        total = sum(list(counts.values()))
        for est_num_spk, count in counts.items():
            speaker_count[nmix][
                est_num_spk
            ] = f"{count}  {round(100*count/total, 2)}[%]"
    return speaker_count


def write_results(
    output_dir, results, max_num_spk, filename="score_summary.json"
):
    for protocol, scores in results.items():
        for nmix, score in scores.items():
            output_dir_nmix = output_dir / f"{nmix}mix_results"
            output_dir_nmix.mkdir(exist_ok=True)
            if protocol != "Est_num_spk":
                with open(output_dir_nmix / f"{protocol}.txt", "w") as f:
                    f.write(score)
            else:
                with open(output_dir_nmix / f"{protocol}.json", "w") as f:
                    json.dump(score, f)
    with open(output_dir / filename, "w") as f:
        json.dump(results, f, indent=4)


def get_parser():
    parser = config_argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--protocols", type=str, required=True)
    parser.add_argument("--max_num_spk", type=int, required=True)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.protocols = args.protocols.split(" ")
    results = score_summary(args)
    write_results(
        args.output_dir,
        results,
        args.max_num_spk,
        filename="score_summary.json",
    )

    # since 4- and 5-mix of wsj0-mix sometimes uses utterances from the same speaker to make mixtures
    # we obtain results with and without speaker duplication
    if args.max_num_spk >= 4:
        for filter_type in ["with_duplication", "without_duplication"]:
            results = score_summary(args, filter_type=filter_type)
            write_results(
                args.output_dir,
                results,
                args.max_num_spk,
                filename=f"score_summary_{filter_type}.json",
            )


if __name__ == "__main__":
    main()
