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
