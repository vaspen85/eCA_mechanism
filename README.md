# Enhanced Cascading Architecture (eCA)

This repository provides a PyTorch implementation of the enhanced Cascading Architecture (eCA), designed to improve image classification accuracy through class-specific early exits. It includes tools to train class routers (`Bi`), refine selected classes (`Ri`), and construct and deploy the full inference system based on algorithms from the original paper.

## 🔧 Project Structure

```plaintext
eCA_repo/
│
├── train_Bi.py           # Train class-specific binary routers and store class activation vectors (ρi)
├── train_Ri.py           # Train refinement classifiers for selected classes using stored ρi vectors
├── select_classes.py     # Implements Algorithm 3 to select which classes to cascade
├── inference_eCA.py      # Implements Algorithm 4 to perform inference using the eCA design
├── utils.py              # Shared utility functions (e.g., accuracy, dataset loading)
├── models/               # Folder for storing trained models (Bi, Ri, Nb)
├── data/                 # Optional placeholder for dataset paths or metadata
└── README.md             # This file
````

---

## 📘 Requirements

* Python 3.7+
* PyTorch
* torchvision
* NumPy
* tqdm

Install the requirements with:

```bash
pip install torch torchvision numpy tqdm
```

---

## 🚀 Usage

### 1. Train Class Routers (Bi) and Store ρi

```bash
python train_Bi.py --dataset CIFAR100 --model resnet18
```

This trains a binary router `Bi` for each class and stores its confidence vector `ρi` (probability of selecting each class) in a `.pt` file.

---

### 2. Train Refiners (Ri) for Each Class

```bash
python train_Ri.py --dataset CIFAR100 --model resnet18 --rho_dir ./rho_vectors/
```

This uses the stored `ρi` vectors to define the subset of classes each `Ri` should specialize in and trains them accordingly.

---

### 3. Select Classes for eCA (Algorithm 3)

```bash
python select_classes.py --dataset CIFAR100 --model resnet18
```

Implements Algorithm 3 to determine which classes benefit from cascading and should be handled by `Bi`–`Ri` pairs.

---

### 4. Run Inference with Full eCA Architecture (Algorithm 4)

```bash
python inference_eCA.py --dataset CIFAR100 --model resnet18
```

Performs inference by routing test images through Bi-Ri pairs or to the baseline classifier `Nb` as per Algorithm 4.

---

## 📊 Results

The eCA mechanism significantly improves per-class accuracy for selected low-accuracy classes and reduces inference time through early exits. See the original paper for benchmark results on CIFAR-10, CIFAR-100, Tiny-ImageNet, MNIST, and ImageNet-500.

---

## 📝 Citation

If you use this code in your research, please cite:

> Vasileios Pentsos et al., "Improved Image Classification using Lightweight Deep Neural Network Enhancements", ACM TIST, 2025 (under revision)

---

## 📫 Contact

For questions or contributions, please contact:
[vaspen85@gmail.com]
```
