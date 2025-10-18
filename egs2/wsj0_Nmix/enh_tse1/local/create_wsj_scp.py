from pathlib import Path
import glob
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
FS_ORIG = 16000
MAX_NUM_SPK = 5

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--wsj0_path", default="../")
parser.add_argument("-o", "--output_folder", default="wsj0-mix")
parser.add_argument("-n", "--n_src", default=2, type=int)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
parser.add_argument("--len_mode", nargs="+", type=str, default=["min", "max"])
parser.add_argument("--metadata_output_folder", type=Path, default="./metadata2")
parser.add_argument("--scp_output_dir", type=Path, required=True)
args = parser.parse_args()


args.metadata_output_folder.mkdir(exist_ok=True)
for cond in ["tr", "cv", "tt"]:
    # Output folders (wav8k-16k/min-max/tr-cv-tt/mix-src{i})
    base = Path(args.output_folder) / f"{args.n_src}speakers" / f"wav{args.samplerate // 1000}k"
    min_mix_folder =  base / "min"/  cond / "mix"
    min_src_folders = [base / "min"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    max_mix_folder =  base / "max"/  cond / "mix"
    max_src_folders = [base / "max"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    for p in min_src_folders + max_src_folders + [min_mix_folder, max_mix_folder]:
        p.mkdir(parents=True, exist_ok=True)

    # Read SNR scales file
    header = [x for t in zip([f"s_{i}" for i in range(args.n_src)], [f"snr_{i}" for i in range(args.n_src)]) for x in t]
    mix_df = pd.read_csv(f"metadata/mix_{args.n_src}_spk_{cond}.txt", delimiter=" ", names=header, index_col=False)

    # make scp output folder
    scp_output_folder = {}
    for len_mode in args.len_mode:
        # output folders
        scp_output_folder[len_mode] = args.scp_output_dir / f"{cond}_{len_mode}_{args.samplerate // 1000}k"
        scp_output_folder[len_mode].mkdir(exist_ok=True)

        # prepare writers
        mode = "w" if not (scp_output_folder[len_mode] / "wav_org.scp").exists() else "a"
        mix_metadata_writer = open(args.metadata_output_folder / f"mix_{args.n_src}_spk_{len_mode}_{cond}_mix", mode, encoding="UTF-8")
        wav_scp_writer = open(scp_output_folder[len_mode] / "wav_org.scp", mode, encoding="UTF-8")
        utt2spk_writer = open(scp_output_folder[len_mode] / "utt2spk_org", mode, encoding="UTF-8")
        spk_metadata_writers = []
        spk_scp_writers = []
        for i in range(args.n_src):
            spk_metadata_writers.append(open(args.metadata_output_folder / f"mix_{args.n_src}_spk_{len_mode}_{cond}_{i}", mode, encoding="UTF-8"))
        for i in range(MAX_NUM_SPK):
            spk_scp_writers.append(open(scp_output_folder[len_mode] / f"spk{i+1}_org.scp", mode, encoding="UTF-8"))

        # start writing scp files
        for idx in tqdm(range(len(mix_df))):
            # Merge filenames for mixture name.  (when mixing weight is 0.450124, it truncates 0.45012, hence the 10x)
            pp = lambda x: x.split('/')[-1].replace(".wav", "") if isinstance(x, str) else '{:12.8g}'.format(x).strip()
            filename = "_".join([pp(mix_df[u][idx]) for u in header]) + ".wav"

            # write the filename of wavs
            fname = filename.replace(".wav", "")
            mix_metadata_writer.write(f"{fname}\n")
            # write the path of original files
            for i in range(args.n_src):
                p = str(Path(args.wsj0_path) / mix_df[f"s_{i}"][idx])
                spk_metadata_writers[i].write(f"{p}\n")

            # make speaker ids
            folder = min_mix_folder if len_mode == "min" else max_mix_folder
            spk_ids = [] # speaker IDs
            for i in range(args.n_src):
                spk_ids.append(filename.split("_")[i*2][:3])
            spk_ids = '_'.join(spk_ids)

            # write wav_org.scp
            wav_scp_writer.write(f"{args.n_src}mix_{spk_ids}_{fname} {str(folder/filename)}\n")

            # write spk?_org.scp
            folder = min_src_folders if len_mode == "min" else max_src_folders
            for i in range(1, MAX_NUM_SPK+1, 1):
                if args.n_src < i:
                    content = f"{args.n_src}mix_{spk_ids}_{fname} dummy\n"
                else:
                    content = f"{args.n_src}mix_{spk_ids}_{fname} {str(folder[i-1]/filename)}\n"
                spk_scp_writers[i-1].write(content)

            # write utt2spk file
            utt2spk_writer.write(f"{args.n_src}mix_{spk_ids}_{fname} {args.n_src}mix_{spk_ids}\n")

        # close writers
        mix_metadata_writer.close()
        wav_scp_writer.close()
        utt2spk_writer.close()
        for i in range(args.n_src):
            spk_metadata_writers[i].close()
        for i in range(MAX_NUM_SPK):
            spk_scp_writers[i].close()
