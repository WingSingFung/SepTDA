# SepTDA: Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor

This is an unofficial reproduction of **SepTDA** from the ICASSP 2024 paper:

📄 [Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor](http://arxiv.org/abs/2401.12473)

---

## Reproduction Results

∆SI-SDR (dB) comparison on WSJ0-{2,3,4,5}mix. Numbers in parentheses represent speaker counting accuracy (%).

| Model                  | 2-mix         | 3-mix        | 4-mix        | 5-mix         |
| ---------------------- | ------------- | ------------ | ------------ | ------------- |
| SepTDA (paper) [2−5]  | 23.6          | 23.5         | 22.0         | 21.0          |
| SepTDA (paper)* [2−5] | 23.6 (99.90)  | 22.1 (95.93) | 19.5 (90.10) | 16.9 (83.23)  |
| SepTDA (ours) [2−5]   | 23.92         | 23.91        | 22.25        | 20.98         |
| SepTDA (ours)* [2−5]  | 23.84 (98.47) | 22.69 (95.3) | 20.4 (91.73) | 16.57 (81.63) |

\* indicates results with speaker counting prediction (unknown number of speakers scenario).

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Environment Setup

Modify line 2 in `egs2/wsj0_Nmix/enh_tse1/enh_tse.sh` to set the ESPnet path:

```bash
export PYTHONPATH="/path/to/your/SepTDA:$PYTHONPATH"
```

---

## Data Preparation

### Prerequisites

**Important**: The WSJ0 dataset requires a commercial license and must be downloaded separately.

### Setup Instructions

1. Navigate to the experiment directory:

   ```bash
   cd egs2/wsj0_Nmix/enh_tse1
   ```
2. Choose one of the following options based on your WSJ0 dataset format:

#### Option A: WSJ0 in WAV Format

If your WSJ0 dataset is already in WAV format, create a symbolic link:

```bash
mkdir -p ./data/wsj0
ln -s /path/to/your/WSJ0 ./data/wsj0/wsj0
```

Alternatively, modify line 24 in `./local/data.sh`:

```bash
wsj_full_wav=/path/to/your/WSJ0/
```

#### Option B: WSJ0 in Original Format (sphere/sph)

If your dataset is in the original WSJ0 format:

1. Uncomment lines 76-81 in `./egs2/wsj0_2mix/enh1/local/data.sh`
2. Set the `WSJ0=` path in `db.sh`

### Generate Mixed Data

Configure the following parameters in `./run.sh`:

- `min_or_max`: Minimum or maximum speaker configuration
- `sample_rate`: Audio sample rate (default: 8000 Hz)

Run the data preparation pipeline:

```bash
# Data will be generated in ./data/wsj0_mix/?speakers
./run.sh --stage 1 --stop_stage 5 \
  --enh_args "--task tse" \
  --audio_format wav
```

---

## Training

Train the SepTDA model using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh \
  --stage 6 \
  --stop_stage 6 \
  --enh_config ./conf/ropetda/sepropetda_multidecoderloss_slow.yaml \
  --ngpu 4
```

**Training Parameters:**

- `--ngpu`: Number of GPUs to use (adjust based on your hardware)
- `--enh_config`: Model configuration file

---

## Inference

Perform speaker separation on test data:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh \
  --stage 7 \
  --stop_stage 8 \
  --enh_config ./conf/ropetda/sepropetda_multidecoderloss_slow.yaml \
  --inf_nums "2 3 4 5" \
  --inference_model valid.loss.best.pth
```

**Inference Parameters:**

- `--inf_nums`: Number of speakers to separate (space-separated, e.g., "2 3 4 5")
- `--inference_model`: Model checkpoint to use (e.g., `valid.loss.best.pth` or `last.pth`)

### Output

Evaluation results will be saved to:

```
egs2/wsj0_Nmix/enh_tse1/exp/enh_sepropetda_multidecoderloss_slow_raw/enhanced_tt_min_8k/enh/SISNR_improvement.json
```

---

## Acknowledgments

This repository contains a streamlined version of ESPnet-Enh, designed for easier training and inference of TISDiSS. Since the full ESPnet framework can be complex for new users, we provide this simplified codebase focused specifically on our method. For additional examples, features, and the complete ESPnet-Enh toolkit, please refer to the [ESPnet-Enh repository](https://github.com/espnet/espnet).

Special thanks to [@kohei0209 (Kohei Saijo)](https://github.com/kohei0209) for his excellent work on:

> **"A Single Speech Enhancement Model Unifying Dereverberation, Denoising, Speaker Counting, Separation, and Extraction"**
> 📄 Paper: http://arxiv.org/abs/2310.08277
> 💻 Code: https://github.com/kohei0209/espnet/tree/muse/egs2/wsj0_Nmix

His implementation provided valuable insights for handling unknown-number speaker separation scenarios.

---

## Citation

If you use this code in your research, please cite the original SepTDA paper:

```bibtex
@inproceedings{lee2024boosting,
  title={Boosting unknown-number speaker separation with transformer decoder-based attractor},
  author={Lee, Younglo and Choi, Shukjae and Kim, Byeong-Yeol and Wang, Zhong-Qiu and Watanabe, Shinji},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={446--450},
  year={2024},
  organization={IEEE}
}
```

---

## License

This project is intended for research purposes only. Please ensure compliance with the WSJ0 dataset license terms.
