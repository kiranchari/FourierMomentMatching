
# Fourier Moment Matching

**[TMLR 2023]**  
Code for the paper:  
**_Mitigating Real-World Distribution Shifts in the Fourier Domain_**

This repository contains a reference implementation of Fourier Moment Matching (FMM) for data augmentation and domain adaptation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/da09adc5-557f-42f2-a64b-b044931133ee" width="750px">
</p>

---

## 📁 Files

- **`fmm.py`**: Contains the core `FMM` class responsible for computing spectral statistics (mean, standard deviation, covariance) and performing moment matching in the Fourier domain.
- **`main.py`**: An example script demonstrating how to use the `FMM` class with dummy datasets.

---

## ⚙️ Setup

### 🔧 Prerequisites

- Python 3.x  
- PyTorch  
- NumPy  
- Pillow (`PIL`)  
- [`robustness`](https://github.com/MadryLab/robustness) library (used by `fmm.py` via `robustness.datasets`)

### 📦 Installation

Ensure you have PyTorch installed. Then install the additional dependencies:

```bash
pip install torch torchvision numpy Pillow
```

You must also install the `robustness` library and ensure `sqrtm.py` is correctly placed in your Python path.

---

## 🚀 Usage

The `FMM` class matches the spectral moments (mean, standard deviation, or covariance) of a source dataset to a target dataset.

### 🔍 `FMM` Class Parameters (from `fmm.py`)

| Parameter | Type | Description |
|----------|------|-------------|
| `target_dataset` | `str` | Name of the target dataset (e.g., `'cifar10'`, `'mnist'`) |
| `batch_size` | `int` | Batch size for data loaders |
| `source_loader` | `DataLoader` | DataLoader for the source dataset |
| `source_dataset` | `str`, optional | Name of the source dataset |
| `match_cov` | `bool`, optional | If `True`, matches covariance; otherwise, matches mean/std. Default: `True` |
| `target_loader` | `DataLoader`, optional | Required if `target_dataset` is not built-in |
| `mean_only` | `bool`, optional | If `True`, only matches the mean. Requires `match_cov=False` |
| `ledoit_wolf_correction` | `bool`, optional | Applies Ledoit-Wolf covariance correction |
| `block_diag` | `bool` or `int`, optional | Use block-diagonal approximation for covariance matrices |
| `large_img_sample_size` | `int`, optional | Max number of large images for statistics computation |
| `use_2D_dft` | `bool`, optional | Use 2D DFT instead of 3D DFT |

---

## 🧪 Example (`main.py`)

The `main.py` script demonstrates a basic usage of the `FMM` class. It sets up dummy source and target datasets and applies FMM to a batch of images before any explicit input normalization (beyond `ToTensor`).

---

## 📄 Citation

If you use this code, please cite the following paper:

```bibtex
@article{
krishnamachari2023mitigating,
title={Mitigating Real-World Distribution Shifts in the Fourier Domain},
author={Kiran Krishnamachari and See-Kiong Ng and Chuan-Sheng Foo},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=lu4oAq55iK},
note={}
}
```
