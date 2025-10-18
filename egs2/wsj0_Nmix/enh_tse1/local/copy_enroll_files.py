from pathlib import Path
import argparse
import json

from espnet2.fileio.read_text import read_2columns_text

"""
This code is for rewrite scp files with replacing the specified words with other words.
For example, this code can replace "min" to "max" in the path.
It is useful to make scp files of "max" version of WSJ from "min" version without changing other configuration.

Source: ./data/wsj0_mix/5speakers/wav8k/min/tt/s5/442o030e_1.0120967_420o030c_0.9186296_053o020p_-0.9186296_443c020l_-1.0120967_050c010j_0.wav
Target: ./data/wsj0_mix/5speakers/wav8k/max/tt/s5/442o030e_1.0120967_420o030c_0.9186296_053o020p_-0.9186296_443c020l_-1.0120967_050c010j_0.wav

python ./local/copy_enroll_files.py \
    --source_folder ../enh_tse1/data/tt_min_8k --target_folder ./data/tt_max_8k \
    --source_words /min/ /enh_tse1/ --target_words /max/ /enh_tse1_max/ --num_spk 5
"""

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=Path, required=True)
parser.add_argument("--target_folder", type=Path, required=True)
parser.add_argument(
    "--source_words",
    nargs="*", type=str, required=True,
    help="Specify the word that you would like to replace from"
)
parser.add_argument(
    "--target_words",
    nargs="*", type=str, required=True,
    help="Specify the word that you would like to replace with"
)
parser.add_argument("--num_spk", type=int, default=5)
args = parser.parse_args()


assert len(args.source_words) == len(args.target_words)
# print(args.source_words, args.target_words)

for spk in range(args.num_spk):
    if (args.source_folder / f"enroll_spk{spk + 1}.scp").exists():
        # load enroll_spk?.scp from source folder
        source_file = read_2columns_text(args.source_folder / f"enroll_spk{spk + 1}.scp")
        # open enroll_spk?.scp in target folder to write
        target_file = open(args.target_folder / f"enroll_spk{spk + 1}.scp", "w")
        # rewrite some specified words
        for utt_id, audio_path in source_file.items():
            new_audio_path = audio_path
            for (source_word, target_word) in zip(args.source_words, args.target_words):
                new_audio_path = new_audio_path.replace(source_word, target_word)
            target_file.write(f"{utt_id} {new_audio_path}\n")

        # close the wrriten file
        target_file.close()

# copy spk2enroll file
if (args.source_folder / "spk2enroll.json").exists():
    print("Copy spk2enroll.json")
    # read original spk2enroll file
    with open(args.source_folder / "spk2enroll.json", "r") as f:
        source_file = json.load(f)
    # rewrite some specified words
    new_dict = {}
    for spk_id, data in source_file.items():
        new_dict[spk_id] = []
        for utt_id, audio_path in data:
            new_audio_path = audio_path
            for (source_word, target_word) in zip(args.source_words, args.target_words):
                new_audio_path = new_audio_path.replace(source_word, target_word)
            new_dict[spk_id].append([utt_id, new_audio_path])
    # save a new json file to target folder
    with open(args.target_folder / "spk2enroll.json", "w") as f:
        json.dump(new_dict, f, indent=4)
