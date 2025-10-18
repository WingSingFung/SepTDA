from pathlib import Path
from espnet2.fileio.read_text import read_2columns_text
import argparse

def make_utt2category_file(
    wav_scp, output_dir, prefix="utt2category",
):
    data = read_2columns_text(wav_scp)
    for i, key in enumerate(list(data.keys())):
        mode = "w" if i == 0 else "a"
        with open(output_dir / f"{prefix}", mode, encoding="UTF-8") as f:
            utt2category = f"{key} {key[:4]}" # e.g., key: 2mix_011_013_011c020h_0.9531_013c020l_-0.9531
            f.write(utt2category + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Path to the wav.scp file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the directory for storing output files",
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        default="utt2category",
        help="Prefix of the output files",
    )
    args = parser.parse_args()

    # make utt2category
    make_utt2category_file(
        args.wav_scp,
        args.output_dir,
        prefix=args.outfile_prefix,
    )