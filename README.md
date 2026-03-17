<div align="center">

<h1>DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification</h1>

<a href=""><img src="https://img.shields.io/badge/Paper-SpMR" alt="Paper PDF"></a>
<a href=""><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://dermaflux.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


[Stathis Galanakis](https://stathisgln.github.io/), [Alexandros Koliousis](https://akoliousis.com/),  [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)
</div>

```bibtex
```

## TLDR:
We introduce **DermaFlux**:  <br>
&nbsp;&nbsp;&nbsp; 🔥 **DermaFlux** generates realistic skin lesion images from text using rectified flows, enabling efficient and semantically aligned medical image synthesis.  <br>
&nbsp;&nbsp;&nbsp; 🔥 Trained on a **~500k curated dermatology image–text** dataset with captions describing clinically relevant attributes such as *asymmetry, border irregularity,* and *color variation.*  <br>
&nbsp;&nbsp;&nbsp; 🔥 DermaFlux synthetic data improves classification performance by up to +6% when augmenting small real datasets and +9% compared to diffusion-based synthetic images. <br>

## Quick Start

### Prerequisites


Create a python 3.10 enviroment 

1. Clone the repository and change directory
```bash
git clone https://github.com/dermaflux/dermaflux.git
cd dermaflux
```

2. Create a python 3.10 enviroment and install dependencies

```bash 
conda create -n dermaflux python=3.10
pip install -r requirements.txt
```

### Downloading Required Models

1. Create checkpoint subfolders
```bash
cd checkpoints
mkdir flux
mkdir flux_text_encoders
```

2. To run DermaFlux, you need to download the original Flux 1. checkpoints:
   - **DiT, AE**: Download from the [black-forest-labs/FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) dev repository and place them under the `checkpoints/flux/` directory.
     - Use `flux1-dev.safetensors` and `ae.safetensors`
     - Note: Weights in the subfolder are in Diffusers format and cannot be used
   - **Text Encoder 1 (T5-XXL)**, **Text Encoder 2 (CLIP-L)**: Download from the [ ComfyUI FLUX Text Encoders repository](https://huggingface.co/comfyanonymous/flux_text_encoders) and place them under the `checkpoints/flux_text_encoders/` directory.
     - Use `t5xxl_fp16.safetensors` for T5-XXL and `clip_l.safetensors` for CLIP-L.

   - **Lora weights**: Download them from DermaFLux's [huggingface repository]() and place them under the `checkpoints` directory.

### Run the gradio application
Run the sampling code using the following command:

```bash
python gradio_app.py
```


## Generated dataset
We provide a generated dataset, generated using DermaFlux in this [link]() containing 40k generated benign skin lesion images and 40k malignant skin lesion.

## Acknowledgements

Thanks for the contributions of the following repo: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts/tree/sd3)