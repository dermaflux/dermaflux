import math
from typing import Callable, Optional
import random
from tqdm import tqdm
import numpy as np
import einops
from transformers import CLIPTextModel

from library import flux_models, flux_utils, strategy_flux, device_utils, lora_flux
from safetensors.torch import load_file


import gradio as gr

import torch

def is_fp8(dt):
    return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    neg_txt: Optional[torch.Tensor] = None,
    neg_vec: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    # this is ignored for schnell
    print(f"guidance: {guidance}, cfg_scale: {cfg_scale}")
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # prepare classifier free guidance
    if neg_txt is not None and neg_vec is not None:
        b_img_ids = torch.cat([img_ids, img_ids], dim=0)
        b_txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
        b_txt = torch.cat([neg_txt, txt], dim=0)
        b_vec = torch.cat([neg_vec, vec], dim=0)
        if t5_attn_mask is not None and neg_t5_attn_mask is not None:
            b_t5_attn_mask = torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0)
        else:
            b_t5_attn_mask = None
    else:
        b_img_ids = img_ids
        b_txt_ids = txt_ids
        b_txt = txt
        b_vec = vec
        b_t5_attn_mask = t5_attn_mask

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((b_img_ids.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            b_img = torch.cat([img, img], dim=0)
        else:
            b_img = img

        pred = model(
            img=b_img,
            img_ids=b_img_ids,
            txt=b_txt,
            txt_ids=b_txt_ids,
            y=b_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            txt_attention_mask=b_t5_attn_mask,
        )

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            pred_uncond, pred = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)

        img = img + (t_prev - t_curr) * pred

    return img

def do_sample(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    print(f"num_steps: {num_steps}")
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # denoise initial noise
    with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
        x = denoise(
            model,
            img,
            img_ids,
            t5_out,
            txt_ids,
            l_pooled,
            timesteps,
            guidance,
            t5_attn_mask,
            neg_t5_out,
            neg_l_pooled,
            neg_t5_attn_mask,
            cfg_scale,
        )

    return x

def generate_image(
    model,
    clip_l: CLIPTextModel,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
    offload = False,
    apply_t5_attn_mask=False,
    dtype='bf16'
):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise_dtype = torch.float32 if is_fp8(dtype) else dtype
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=noise_dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # prepare img and img ids
    # txt2img only needs img_ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)


    # prepare fp8 models
    if is_fp8(clip_l_dtype) and (not hasattr(clip_l, "fp8_prepared") or not clip_l.fp8_prepared):
        print(f"prepare CLIP-L for fp8: set to {clip_l_dtype}, set embeddings to {torch.bfloat16}")
        clip_l.to(clip_l_dtype)  # fp8
        clip_l.text_model.embeddings.to(dtype=torch.bfloat16)
        clip_l.fp8_prepared = True

    if is_fp8(t5xxl_dtype) and (not hasattr(t5xxl, "fp8_prepared") or not t5xxl.fp8_prepared):
        print(f"prepare T5xxl for fp8: set to {t5xxl_dtype}")

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)

                    hidden_states = module.wo(hidden_states)
                    return hidden_states

                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    module.forward = forward_hook(module)

        t5xxl.to(t5xxl_dtype)
        prepare_fp8(t5xxl.encoder, torch.bfloat16)
        t5xxl.fp8_prepared = True

    # prepare embeddings
    print("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)

            if is_fp8(t5xxl_dtype):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
    if negative_prompt:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
    else:
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    # del clip_l, t5xxl
    device_utils.clean_memory()

    # generate image
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if apply_t5_attn_mask else None

    x = do_sample(
        model,
        noise,
        img_ids,
        l_pooled,
        t5_out,
        txt_ids,
        steps,
        guidance,
        t5_attn_mask,
        is_schnell,
        device,
        flux_dtype,
        neg_l_pooled,
        neg_t5_out,
        neg_t5_attn_mask,
        cfg_scale,
    )
    if offload:
        model = model.cpu()
    # del model
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    return x


class AppWrapper:    

    def __init__(self, model, clip_l, t5xxl, vae):
        self.model = model
        self.clip_l = clip_l
        self.t5xxl = t5xxl
        self.vae = vae
        
    def generate_image_func(self, prompt, seed=42, image_size=512,
                        steps=20, guidance=3.5, cfg_scale=1.0, 
                        offload=True, negative_prompt=None, ):
                        
        height = width = image_size

        x = generate_image(
                self.model,
                self.clip_l,
                self.t5xxl,
                self.vae,
                prompt,
                seed,
                width,
                height,
                steps,
                guidance,
                negative_prompt,
                cfg_scale,
                dtype=dtype
            )

            
        if offload:
            self.vae = self.vae.to(device)
            
        with torch.no_grad():
            x = vae.decode(x.to(ae_dtype))

        if offload:
            self.vae = self.vae.cpu()
        

        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)
        img = (127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)
        
        return img



## Define device
device = torch.device('cuda:0')

dtype = torch.bfloat16
clip_l_dtype = dtype
t5xxl_dtype = dtype
ae_dtype = dtype
flux_dtype = dtype

## Paths to models and weights
flux_ckpt_path = 'checkpoints/flux/flux1-dev.safetensors'
clip_l_path = 'checkpoints/flux_text_encoders/clip_l.safetensors'
t5xxl_path = 'checkpoints/flux_text_encoders/t5xxl_fp16.safetensors'
ae_path = 'checkpoints/flux/ae.safetensors'

lora_weight = "checkpoints/flux_lora_64_64.safetensors"

# load clip_l
clip_l = flux_utils.load_clip_l(clip_l_path, clip_l_dtype, device)
clip_l.eval()

t5xxl = flux_utils.load_t5xxl(t5xxl_path, t5xxl_dtype, device)
t5xxl.eval()

# DiT
is_schnell, model = flux_utils.load_flow_model(flux_ckpt_path, None, device)
model.eval()
print(f"Casting model to {flux_dtype}")
model.to(flux_dtype)  # make sure model is dtype

t5xxl_max_length = 256 if is_schnell else 512
tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

# AE
vae = flux_utils.load_ae(ae_path, ae_dtype, device)
vae.eval()


height = 512
width = 512

# LoRA
lora_models = []
for weights_file in [lora_weight]:
    if ";" in weights_file:
        weights_file, multiplier = weights_file.split(";")
        multiplier = float(multiplier)
    else:
        multiplier = 1.0
        # multiplier = 0.5
    print(f"Loading LoRA weights from {weights_file} with multiplier {multiplier}")
    weights_sd = load_file(weights_file)

    module = lora_flux 
    lora_model, _ = module.create_network_from_weights(multiplier, None, vae, [clip_l, t5xxl], model, weights_sd, True)

    lora_model.apply_to([clip_l, t5xxl], model)
    info = lora_model.load_state_dict(weights_sd, strict=True)
    lora_model.eval()
    lora_model.to(device)

    lora_models.append(lora_model)


vae = vae.to('cpu')

app_wrapper = AppWrapper(model, clip_l, t5xxl, vae)

torch.cuda.empty_cache()

default_caption = "This photo depicts a malignant lesion. Here's a description of the mole: " +\
        "The mole exhibits a pronounced asymmetry, characterized by an uneven distribution of pigment and shape, with one half being notably darker than the other. " +\
        "The border of the mole is irregular, featuring jagged edges and a lack of smooth, uniform contours. The color of the mole is predominantly brown, with patches of " +\
        "lighter and darker shades scattered throughout, creating a mottled appearance. Overall, the mole displays a concerning combination of features that are often indicative of malignancy."

example_prompts = [
    [   "This photo depicts a malignant lesion. Here's a description of the mole: \n \nThe mole is characterized by a significant degree of asymmetry, with its right side being notably larger than the left. " +\
        "The border of the mole is irregular, featuring a jagged and uneven outline that lacks the smooth, symmetrical edges typically seen in benign moles. Additionally, the mole exhibits a mottled appearance, " +\
        "with a mix of light brown and dark brown colors that are not evenly distributed throughout the lesion."
    ],
    [   "This photo depicts a malignant lesion. Here's a description of the mole: \nThe mole in the image exhibits a combination of features that are concerning for malignancy. The most notable aspect is its asymmetry, " +\
        "with one half of the mole appearing larger than the other. This asymmetry is evident in the shape and size of the mole, with the left side being more irregular and larger than the right side.\n\n" +\
        "The border of the mole is also irregular, with a jagged and notched appearance. This irregularity is particularly noticeable along the left side of the mole, where the border is more defined and has a more pronounced notch. " +\
        "In contrast, the right side of the mole has a smoother border with fewer notches.\n\nThe color of the mole is also abnormal, with a mix of brown and black pigmentation. " +\
        "The mole has a mottled appearance, with areas of darker brown and black pigmentation scattered throughout. There are also some lighter brown areas, particularly on the right side of the mole.\n\n" +\
        "Overall, the combination of asymmetry, irregular border, and abnormal coloration suggests that this mole may be malignant. It is essential to have it evaluated by a dermatologist for further examination and diagnosis."
    ],
    ["This photo depicts a benign lesion. Here's a description of the mole:\n\nThe mole is relatively small, with a diameter of approximately 2-3 millimeters. It is located in the center of the image, " +\
        "and its borders are well-defined and symmetrical. The mole's color is a uniform light brown, with no visible patches or areas of darker or lighter pigmentation. The mole's surface is smooth and " +\
            "flat, with no visible bumps or ridges. Overall, the mole appears to be a typical benign lesion with no signs of malignancy."
    ],
    ["This photo depicts a benign lesion. Here's a description of the mole:\n\nThe mole in the image appears to be symmetrical, with a smooth, even border that does not exhibit any significant irregularities. " +\
        "The color of the mole is uniform and consistent, with a light brown hue that is evenly distributed throughout the lesion. There are no visible areas of darker or lighter pigmentation, and the mole does " +\
            "not appear to have any notable variations in color or texture. Overall, the mole appears to be a benign, symmetrical, and uniformly colored lesion."
    ],
]
    


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## DermaFlux sampling")

    with gr.Row() as row:
        with gr.Column() as col:
            prompt = gr.Textbox(
                label="Description",
                info="Description",
                lines=3,
                value=default_caption,
                interactive=True,
            )
            generate_mask = gr.Button("Generate mask")

            with gr.Accordion("Sampling options", open=False):
                seed = gr.Slider(
                        label="Seed",
                        minimum=1,
                        maximum=10000,
                        step=1,
                        value=42,
                    )
                image_size = gr.Radio([128, 256, 512], value=128)

        with gr.Column() as col:
            gallery = gr.Gallery(label="Generated Image", allow_preview=True, interactive=False)

    inputs = [prompt, seed, image_size]
    generate_mask.click(fn=app_wrapper.generate_image_func, inputs=inputs, outputs=[gallery])
    
    with gr.Row() as row:
        examples = gr.Examples(
            examples=example_prompts,
            inputs=[prompt],
        )
    
if __name__ == '__main__':

    demo.launch()
    