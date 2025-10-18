from pathlib import Path
import argparse

from espnet2.fileio.read_text import read_2columns_text


parser = argparse.ArgumentParser()
parser.add_argument("--wavscp_folder", type=Path, required=True)
parser.add_argument("--output_folder", type=Path, required=True)
parser.add_argument("--nmix", type=int, required=True)
args = parser.parse_args()

file_lists = ["spk2utt", "utt2category", "utt2num_samples", "utt2spk", "wav.scp",]
for n in range(args.nmix):
    file_lists.append(f"enroll_spk{n + 1}.scp")
    file_lists.append(f"spk{n + 1}.scp")
    file_lists.append(f"text_spk{n + 1}")

# scp output folders for each number of speakers
(args.wavscp_folder / f"{args.nmix}mix").mkdir(exist_ok=True)

for file_name in file_lists:
    tgt_file_path = args.output_folder / f"{args.nmix}mix" / file_name
    if tgt_file_path.exists():
        print(f"Target file {tgt_file_path} already exists. Skip process")
        continue
    if not (args.wavscp_folder / file_name).exists():
        print(f"Source file {args.wavscp_folder / file_name} does not exist. Skip process")
        continue
    # open original file
    original_file = read_2columns_text(args.wavscp_folder / file_name)
    # open output file
    output_file = (open(args.output_folder / f"{args.nmix}mix" / file_name, "w"))
    # split wav.scp for each number of speakers
    for utt_id, value in original_file.items():
        nmix = int(utt_id[0])
        if nmix != args.nmix:
            continue
        to_write = f"{utt_id} {value}\n"
        output_file.write(to_write)
    # close output file
    output_file.close()

# make symbolic link to feats_type
if (args.wavscp_folder / "feats_type").exists():
    Path(args.output_folder / f"{args.nmix}mix" / "feats_type").symlink_to("../feats_type")
