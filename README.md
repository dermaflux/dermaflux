<div align="center">

<h1>DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification</h1>

<a href="https://arxiv.org/pdf/2603.16392"><img src="https://img.shields.io/badge/Paper-SpMR" alt="Paper PDF"></a>
<a href="https://arxiv.org/abs/2603.16392"><img src="https://img.shields.io/badge/arXiv-2603.16392-b31b1b" alt="arXiv"></a>
<a href="https://dermaflux.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


[Stathis Galanakis](https://stathisgln.github.io/), [Alexandros Koliousis](https://akoliousis.com/),  [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)
</div>

```bibtex
@misc{galanakis2026dermafluxsyntheticskinlesion,
      title={DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification}, 
      author={Stathis Galanakis and Alexandros Koliousis and Stefanos Zafeiriou},
      year={2026},
      eprint={2603.16392},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16392}, 
}
```

# Overview

**DermaFlux** is a generative framework for synthesizing dermatology images from text descriptions using **rectified flows**.  
It enables **semantically aligned medical image generation** and improves downstream classification performance.

### Key Highlights

🔥 **Text-to-image lesion generation**  
DermaFlux generates realistic skin lesion images from textual descriptions.

🔥 **Large-scale dermatology dataset**  
Trained on **~500k curated dermatology image–text pairs**, with captions describing clinically relevant attributes such as:
- asymmetry  
- border irregularity  
- color variation  

🔥 **Improved classification performance**

Using synthetic DermaFlux data:

- **+6%** improvement when augmenting small real datasets  
- **+9%** improvement compared to diffusion-based synthetic images


# Quick Start


##  1. Clone the repository and change directory
```bash
git clone https://github.com/dermaflux/dermaflux.git
cd dermaflux
```

## 2. Create a Python environment and install dependencies:

```bash 
conda create -n dermaflux python=3.10
conda activate dermaflux

pip install -r requirements.txt
```

## 3. Download Required Models

DermaFlux requires the FLUX.1 base checkpoints, text encoders, and DermaFlux LoRA weights.

### 3.1 Create checkpoint subfolders
```bash
cd checkpoints
mkdir flux
mkdir flux_text_encoders
```
Your directory should look like:
```bash
checkpoints/
 ├── flux/
 ├── flux_text_encoders/
```
### 3.2 Download FLUX.1-dev Models

Download from the official  [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) repository and place the following files inside `checkpoints/flux/`

Required files:

-   `flux1-dev.safetensors` (DiT model)
-   `ae.safetensors` (Autoencoder)

⚠️ **Note:** The weights in the subfolder are in Diffusers format and **cannot be used**.
   
### 3.3  Download Text Encoders
Download from the [ ComfyUI FLUX Text Encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) repository and place them under the `checkpoints/flux_text_encoders/` directory.

Required files:

 - `t5xxl_fp16.safetensors`  (T5-XXL)
 - `clip_l.safetensors` (CLIP-L)

### 3.4 Download DermaFlux LoRA Weights
  
Download the LoRA weights from the [**DermaFlux HuggingFace**](https://huggingface.co/StathisGln/DermaFlux) repository
and place them inside `checkpoints`.

# Run the Demo

Launch the **Gradio interface**:

``` bash
python gradio_app.py
```

This will start a local interface for **text-to-skin-lesion
generation**.


---


# Training Dataset


## Download Captions Dataset

The repository includes a captions archive tracked with **Git LFS**:

```bash
git lfs install
git lfs pull
```

After pulling the LFS files, the repository structure should contain:

```text
data/
└── captions.zip
```

The `captions.zip` archive contains the **generated text captions and metadata files** used to train DermaFlux. It does **not** include the underlying dermatology images, which remain subject to the licensing and terms of their respective datasets.


## Download Source Datasets and Corresponding Caption Files

To reproduce the DermaFlux training data, please download the original image datasets from their official sources and pair them with the corresponding `.json` caption files provided in `captions.zip`.


| Dataset     |    Download Link    |  File Name   |
|-------------|---------------------|--------------|
| MedNode | [Download](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/complete_mednode_dataset.zip) | `mednode.json` |
| HIBA | [Download](https://api.isic-archive.com/doi/hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022/) | `hiba.json` |
| Derm12345 | [Download](https://github.com/abdurrahimyilmaz/derm12345) | `derm12345.json`|
| ISIC 2019 | [Download](https://challenge.isic-archive.com/data/#2019) | `isic2019.json` | 
| ISIC 2020 | [Download](https://challenge.isic-archive.com/data/#2020) | `isic2020.json` |
| Milk10k | [Download](https://challenge.isic-archive.com/data/#milk10k) | `milk10k` | 
| PAD20 | [Download](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | `pad20.json` |
| Kaggle 1 | [Download](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | `kaggle1.json` |
| Kaggle 2 | [Download](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data) | `kaggle2.json` |
| DDI | [Download](https://ddi-dataset.github.io/index.html#dataset) | `ddi.json` |
| ISIC 2024 | [Download](https://challenge.isic-archive.com/data/#2024) | `isic2024.json` |



**Note:** We distribute only the generated captions and associated metadata. Users are responsible for obtaining the original datasets and complying with their respective licenses and terms of use.


---

# Synthetic Dermatology Dataset using DermaFlux 

We release a synthetic [dataset](https://huggingface.co/datasets/StathisGln/DermaFlux_synthetic_dataset) generated with **DermaFlux** consisting of:

-   **20k benign skin lesion images**
-   **20k malignant skin lesion images**




## Acknowledgements

This project builds upon the following repository: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts/tree/sd3)
