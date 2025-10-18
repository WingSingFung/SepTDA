# SepTDA: Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor

This is an unofficial reproduction of **SepTDA** from the ICASSP 2024 paper:
📄 [Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor](http://arxiv.org/abs/2401.12473)

---

## Quick Start

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Modify line 2 in `egs2/wsj0_Nmix/enh_tse1/enh_tse.sh` for the espnet path

```
export PYTHONPATH="path/to/your/SepTDA:$PYTHONPATH"
```


---

## Data Preparation

### Prerequisites

**Note**: You need to download the WSJ0 dataset separately (commercial license required).

### Setup Instructions

1. Change to the experiment directory:

   ```bash
   cd ./egs2/wsj0_Nmix/enh_tse1
   ```
2. Choose one of the following options based on your WSJ0 dataset format:

#### Option A: WSJ0 in WAV Format

If your WSJ0 dataset is already in WAV format, create a symbolic link:

```bash
mkdir -p ./data/wsj0
ln -s /path/to/your/WSJ0 ./data/wsj0/wsj0
```

Alternatively, modify line 24 in `./local/data.sh` to point to your WSJ0 path:

```bash
wsj_full_wav=/path/to/your/WSJ0/
```

#### Option B: WSJ0 in Original Format

If your dataset is in the original WSJ0 format:

1. Uncomment lines 76-81 in `./egs2/wsj0_2mix/enh1/local/data.sh`
2. Fill in the `WSJ0=` path in `db.sh`

### Generate Data

Configure the following parameters in `./run.sh` according to your needs:

- `min_or_max`: Minimum or maximum speaker configuration
- `sample_rate`: Audio sample rate

Run data preparation and statistics collection:

```bash
# Data will be generated under ./data/wsj0_mix/?speakers
./run.sh --stage 1 --stop_stage 5 --enh_args "--task tse" --audio_format wav
```

---

## Training

Train the model with the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 6 --stop_stage 6 \
  --enh_config ./conf/ropetda/sepropetda_multidecoderloss_slow.yaml \
  --ngpu 4
```

---

## Inference

Perform speaker separation inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 7 --stop_stage 8 \
  --enh_config ./conf/ropetda/sepropetda_multidecoderloss_slow.yaml \
  --inf_nums "2 3 4 5" \
  --inference_model valid.loss.best.pth
```

**Parameters:**

- `inf_nums`: Number of speakers to separate (e.g., "2 3 4 5")
- `inference_model`: Path to the trained model checkpoint

---

## Acknowledgments

Special thanks to [@kohei0209 (Kohei Saijo)](https://github.com/kohei0209) for his excellent work on:

**"A Single Speech Enhancement Model Unifying Dereverberation, Denoising, Speaker Counting, Separation, and Extraction"**
📄 http://arxiv.org/abs/2310.08277
💻 https://github.com/kohei0209/espnet/tree/muse/egs2/wsj0_Nmix

His work provided valuable reference for unknown-number speaker separation implementation.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{septda2024,
  title={Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor},
  booktitle={ICASSP 2024},
  year={2024}
}
```
