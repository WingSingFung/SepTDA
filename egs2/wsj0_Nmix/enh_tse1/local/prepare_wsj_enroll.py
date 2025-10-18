import json
import random
from itertools import chain
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def prepare_wsjmix_enroll(
    wav_scp, spk2utts, output_dir, num_spk=5, train=True, prefix="enroll_spk"
):
    mixtures = []
    with Path(wav_scp).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mixtureID = line.strip().split(maxsplit=1)[0]
            mixtures.append(mixtureID)

    with Path(spk2utts).open("r", encoding="utf-8") as f:
        # {spkID: [(uid1, path1), (uid2, path2), ...]}
        spk2utt = json.load(f)

    with DatadirWriter(Path(output_dir)) as writer:
        for mixtureID in mixtures:
            # 5mix_40p_40o_01l_01k_20d_40pc020l_1.6040832_40oc0203_0.20397526_01lo030g_-0.20397526_01ka010w_-1.6040832_20do010d_0
            real_num_spk = int(mixtureID[0])
            uttIDs = mixtureID.split("_")[real_num_spk+1:]
            uttIDs = [uttIDs[2*i] for i in range(real_num_spk)]
            assert len(uttIDs) == real_num_spk
            for spk in range(num_spk):
                if spk >= real_num_spk:
                    uttID = "dummy"
                    spkID = "dummy"
                else:
                    uttID = uttIDs[spk]
                    spkID = uttID[:3] # 40pc020l -> 40p
                if train:
                    # For training, we choose the auxiliary signal on the fly.
                    # Thus, here we use the pattern f"*{uttID} {spkID}" to indicate it.
                    writer[f"{prefix}{spk + 1}.scp"][mixtureID] = f"*{uttID} {spkID}"
                else:
                    if spk < real_num_spk:
                        spkID2, enrollID = random.choice(spk2utt[spkID])
                        assert spkID == spkID2[:3], (spkID, spkID2[:3])
                        while enrollID == uttID and len(spk2utt[spkID]) > 1:
                            enrollID = random.choice(spk2utt[spkID])[1]
                    else:
                        enrollID = "dummy"
                    writer[f"{prefix}{spk + 1}.scp"][mixtureID] = enrollID


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Path to the wav.scp file",
    )
    parser.add_argument(
        "spk2utts",
        type=str,
        help="Path to the json file containing mapping from speaker ID to utterances",
    )
    parser.add_argument(
        "--num_spk",
        type=int,
        default=5,
        choices=(2, 3, 4, 5),
        help="Number of speakers in each mixture sample",
    )
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="Whether is the training set or not",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory for storing output files",
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        default="enroll_spk",
        help="Prefix of the output files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    prepare_wsjmix_enroll(
        args.wav_scp,
        args.spk2utts,
        args.output_dir,
        num_spk=args.num_spk,
        train=args.train,
        prefix=args.outfile_prefix,
    )
