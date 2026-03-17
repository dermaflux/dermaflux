<div align="center">

<h1>DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification</h1>

<a href=""><img src="https://img.shields.io/badge/Paper-SpMR" alt="Paper PDF"></a>
<a href=""><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://dermaflux.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


[Stathis Galanakis](https://stathisgln.github.io/), [Alexandros Koliousis](https://akoliousis.com/),  [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)
</div>

```bibtex
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
  
Download the LoRA weights from the [**DermaFlux HuggingFace**]() repository
and place them inside `checkpoints`.

# Run the Demo

Launch the **Gradio interface**:

``` bash
python gradio_app.py
```

This will start a local interface for **text-to-skin-lesion
generation**.

---

# Generated Dataset

We release a synthetic [dataset]() generated with **DermaFlux** consisting of:

-   **40k benign skin lesion images**
-   **40k malignant skin lesion images**





## Acknowledgements

This project builds upon the following repository: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts/tree/sd3)