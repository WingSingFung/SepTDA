from pathlib import Path
from tqdm import tqdm
import argparse

from espnet2.fileio.read_text import read_2columns_text


parser = argparse.ArgumentParser()
parser.add_argument("--transcription_folder", type=Path)
parser.add_argument("--wavscp_folder", type=Path)
parser.add_argument("--num_spk", type=int, default=5)
args = parser.parse_args()

transcriptions = {}
filenames = ["si_tr_s.txt", "si_et_05.txt", "si_dt_05.txt"]
for filename in filenames:
    t = read_2columns_text(args.transcription_folder / filename)
    transcriptions = dict(**transcriptions, **t)

# open wav.scp to get utterance ids
wav_scp = read_2columns_text(args.wavscp_folder / "wav.scp")

# open files to write transcriptions
output_text_files = []
for spk in range(args.num_spk):
    output_text_files.append(open(args.wavscp_folder / f"text_spk{spk + 1}", "w"))

# write transcriptions
for utt_id in tqdm(wav_scp):
    # utt_id is like "2mix_011_021_011o0302_1.959_021o030r_-1.959"
    # first charactor must show number of sources
    nmix = int(utt_id[0])
    # spk_ids are like ["011o0302", "021o030r", ...] for any number of speakers
    spk_ids = utt_id.split("_")[nmix + 1:][::2]
    assert len(spk_ids) == nmix, (nmix, utt_id, spk_ids)

    for spk in range(args.num_spk):
        if spk < nmix:
            text = transcriptions[spk_ids[spk]]
        else:
            text = "dummy"
        to_write = f"{utt_id} {text}\n"
        output_text_files[spk].write(to_write)

# save the files
for spk in range(args.num_spk):
    output_text_files[spk].close()
