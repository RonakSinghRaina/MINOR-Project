# Kestrel: flagging radio frequency interference (RFI) in radio spectrograms

Minor project, School of Data Science, IISER Thiruvananthapuram.
Ronak Singh (IMS23319). Supervisor: Dr. Alwin Poulose.

Radio telescopes record very faint signals from space, but phones, satellites and
aircraft use the same frequencies. This interference (RFI) has to be flagged pixel by
pixel. Kestrel is a small U-Net-based network (593,842 parameters) that does this.
It is compared with `tf_unet` (Akeret et al. 2017) and with the methods published by
Mesarcik et al. (2022), on real LOFAR data checked against a human expert.

This repository holds the code and results behind the main results of the project.

## Main results

Max F1 on the 109 expert-labelled LOFAR test images (mean ± sd over 3 runs):

| Method | max F1 |
|---|---|
| Kestrel | 0.6603 ± 0.0040 |
| Kestrel + expert labels of about 68 images (5-fold) | 0.6747 ± 0.0083 |
| tf_unet | 0.5482 ± 0.0139 |
| RFI-Net (published by Mesarcik et al.) | 0.5979 ± 0.0012 |
| AOFlagger (published by Mesarcik et al.) | 0.5698 |

The scores of every run are in `runs/` (one `metrics.json` per run).

## What is in each folder

| Path | What it is |
|---|---|
| `src/hybrid_rfi_package/hybrid_model.py` | **Kestrel**, the network itself |
| `src/hybrid_rfi_package/train_hybrid.py` | Training helpers used for the synthetic-data run |
| `src/hybrid_rfi_package/hybrid_crf_model.py` | Kestrel with a CRF added on top |
| `src/hybrid_rfi_package/dataset_generator_v3_strength.py` | Makes the synthetic dataset (276 × 600 images) |
| `src/unet_rfi_package copy/tf_unet/` | The authors' unchanged `tf_unet` code (the baseline), with its licence |
| `lofar_data.py`, `hera_data.py` | Load the LOFAR and HERA data quickly, without filling the memory |
| `experiments/lofar_hybrid.py` | Train and test Kestrel on LOFAR or HERA |
| `experiments/lofar_tfunet_baseline.py` | Train and test `tf_unet` on LOFAR or HERA |
| `experiments/mesarcik_repro/` | Mesarcik et al.'s U-Net, and Kestrel under their training settings |
| `experiments/models_ablation.py` | Kestrel with parts that can be switched off (ablation) |
| `experiments/lofar_hybrid_crf.py` | The CRF test |
| `experiments/active_learning/al_lofar.py` | Fine-tuning with expert labels, 5-fold |
| `experiments/baseline_fixednorm.py`, `experiments/width_sweep/` | The synthetic-data runs |
| `scripts/` | One command per experiment; each can be stopped and restarted |
| `analysis/lofar_analysis/` | Finds the test-image copies in the training set, and the σ-clipping score |
| `analysis/hera_aoflagger/` | Builds the HERA data, runs AOFlagger on it, and makes the HERA table |
| `analysis/hera_vs_lofar/` | How good AOFlagger's labels are on LOFAR and HERA |
| `analysis/last_vs_best/` | `tf_unet` scored at its last epoch ("closest to their settings") |
| `analysis/timing/` | Speed: time per spectrogram for each model |
| `analysis/summarise_*.py`, `analysis/active_learning/summarise_al.py` | Print the ablation, no-Dice and fine-tuning tables |
| `runs/` | Scores (`metrics.json`) and training logs of the runs behind the main results |
| `report/Images/Figures/make_data_figures.py` | A helper that one analysis script imports |

Trained model weights are not included (they are large); every run can be repeated
with the scripts.

## Which script gives which result

| Result | Script | Scores in |
|---|---|---|
| Kestrel on LOFAR | `scripts/run_lofar_hybrid.sh` | `runs/lofar/hybrid_b08_nocw_seed*` |
| tf_unet on LOFAR | `scripts/run_lofar_lr1e-4.sh` | `runs/lofar/lr1e-4_seed*` |
| Mesarcik et al.'s U-Net, rerun | `scripts/run_mesarcik_repro.sh` | `runs/lofar/mesarcik_unet_clean_seed0` |
| Kestrel with their settings | `scripts/run_hybrid_mesarcik.sh` | `runs/lofar/hybrid_mesarcik_clean_bce_seed0` |
| Fine-tuning with expert labels | `scripts/run_lofar_al.sh` | `runs/lofar/al/summary.json` |
| Removing parts of Kestrel | `scripts/run_lofar_ablation.sh` | `runs/lofar/abl_*` |
| Kestrel without the Dice loss | `scripts/run_lofar_nodice.sh` | `runs/lofar/nodice_b08_nocw_seed*` |
| All HERA results | `scripts/run_hera.sh` | `runs/hera/` |

## How to run

1. Get the data. LOFAR: `LOFAR_Full_RFI_dataset.pkl` from Mesarcik et al. (2022),
   placed in `data/lofar/`, then run `analysis/lofar_analysis/convert_pickle_to_memmap.py`
   once. HERA: `HERA_04-03-2022_all.pkl`, placed in `data/hera/`, then run
   `analysis/hera_aoflagger/make_hera_arrays.py` once.
2. Make two Python environments, because PyTorch and TensorFlow need different CUDA
   libraries: `requirements-torch.txt` (Kestrel and the analysis) and
   `requirements-tf.txt` (`tf_unet`).
3. Run a script from `scripts/`, for example `./scripts/run_lofar_hybrid.sh`.

Hardware used: NVIDIA RTX 3060 Laptop GPU (6 GB). One Kestrel run on LOFAR takes
about 25 minutes; one `tf_unet` run about 2 h 40 min.

## Licence

GPL-3.0 (see `LICENSE`). `tf_unet` (Akeret et al.) and the Mesarcik et al. U-Net code
keep their own licences, included in their folders.
