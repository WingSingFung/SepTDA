#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

wsj_full_wav=$PWD/data/wsj0/
wsj_mix_wav=$PWD/data/wsj0_mix
wsj_mix_scripts=$PWD/data/wsj0_mix/scripts
wsj_scp_output_dir=$PWD/data


other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt
min_or_max=min
sample_rate=8k

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

# if [ ! -e "${WSJ0}" ]; then
#     log "Fill the value of 'WSJ0' of db.sh"
#     exit 1
# fi
# if [ ! -e "${WSJ1}" ]; then
#     log "Fill the value of 'WSJ1' of db.sh"
#     exit 1
# fi

train_set="tr_"${min_or_max}_${sample_rate}
train_dev="cv_"${min_or_max}_${sample_rate}
recog_set="tt_"${min_or_max}_${sample_rate}


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Prepare simulation scripts and convert WSJ0 to wav"
    echo "Downloading WSJ0_mixture scripts."

    if [ ! -e ${wsj_mix_scripts} ]; then
        mkdir -p ${wsj_mix_scripts}
        git clone https://github.com/mpariente/pywsj0-mix.git ${wsj_mix_scripts}
        cp local/create_wsj_scp.py ${wsj_mix_scripts}

        # remove original script and copy multi-thred version to speed-up data preparation
        rm ${wsj_mix_scripts}/generate_wsjmix.py
        cp local/generate_wsjmix.py ${wsj_mix_scripts}
    else
        echo "${wsj_mix_scripts} already exists. Skip downloading the simulation script."
    fi

    # if [ ! -e ${wsj_full_wav} ]; then
    #     echo "WSJ0 wav file."
    #     local/convert2wav.sh ${WSJ0} ${wsj_full_wav} || exit 1;
    # else
    #     echo "${wsj_full_wav} already exists. Skip the process to convert wsj0 to wav."
    # fi

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data simulation"
    for nsrc in 2 3 4 5; do
        # create mixtures
        local/wsj0_create_mixture.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
            ${wsj_mix_scripts} ${WSJ0} ${wsj_full_wav} \
            ${wsj_mix_wav} ${nsrc} || exit 1;
        # create scp files (unsorted yet)
        local/wsj0_create_scp.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
            ${wsj_mix_scripts} ${WSJ0} ${wsj_full_wav} \
            ${wsj_mix_wav} ${nsrc} ${wsj_scp_output_dir} || exit 1;
    done
    # sort scp files and create utt2category files
    # also prepare transcriptions for max version for ASR evaluation
    local/wsj0mix_data_prep.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
        ${wsj_mix_wav}/wav${sample_rate}/${min_or_max} ${wsj_mix_scripts} ${wsj_full_wav} || exit 1;
fi

# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     log "stage 2: Prepare language information"
#     ### Also need wsj corpus to prepare language information
#     ### This is from Kaldi WSJ recipe
#     log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
#     local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
#     log "local/wsj_format_data.sh"
#     local/wsj_format_data.sh
#     log "mkdir -p data/wsj"

#     if [ ! -e "data/wsj" ]; then
#         mkdir -p data/wsj
#         log "mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj"
#         mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj
#     fi

#     if [ ! -e ${other_text} ]; then
#         log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
#         mkdir -p "$(dirname ${other_text})"

#         # NOTE(kamo): Give utterance id to each texts.
#         zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
#             grep -v "<" | tr "[:lower:]" "[:upper:]" | \
#             awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}
#     fi

#     if [ ! -e ${nlsyms} ]; then
#         log "Create non linguistic symbols: ${nlsyms}"
#         cut -f 2- data/wsj/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
#         cat ${nlsyms}
#     fi
# fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare WSJMix enroll utterances"

    for task in tr cv tt;
    do
        # copy scp files from local while fixing some infomation
        # e.g., min->max, 8k->16k and make them absolute paths
        python local/copy_enroll_files.py \
            --source_folder ./local/enroll_scp/${task}_min_8k \
            --target_folder ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate} \
            --source_words ./ /min/ /wav8k/ \
            --target_words ${PWD}/ /${min_or_max}/ /wav${sample_rate}/

        # for training set, we prepare scp files here
        if [ "$task" = "tr" ]; then
            python local/prepare_wsj_enroll.py \
                ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate}/wav.scp \
                ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate}/spk2enroll.json \
                --num_spk 5 --train true \
                --output_dir ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate} || exit 1;
        fi

    done
fi
