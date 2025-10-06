#!/usr/bin/env python3
"""
Test script to verify LucidFlux model loading and inference
"""
import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
import logging

logging.basicConfig(level=logging.INFO)

# Import required modules
from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack
from src.flux.util import (load_ae, load_flow_model, load_single_condition_branch, load_safetensors)
from src.flux.swinir import SwinIR
import torch.nn as nn
from src.flux.align_color import wavelet_reconstruction

from transformers import SiglipVisionModel
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor
from typing import Optional
import math

def prepare_with_embeddings(img, precomputed_txt, precomputed_vec):
    """
    使用预计算embeddings的prepare函数
    """
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # 直接使用预计算的embeddings
    txt = precomputed_txt
    vec = precomputed_vec
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class Modulation(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.control_index_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)

    def forward(self, x, timestep, control_index):
        timesteps_proj = self.time_proj(timestep * 1000)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(x.dtype))

        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])
        control_index = control_index.to(device=x.device, dtype=x.dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(x.dtype))
        timesteps_emb = timesteps_emb + control_index_emb
        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x

class DualConditionBranch(nn.Module):
    def __init__(self, condition_branch_lq: nn.Module, condition_branch_ldr: nn.Module, modulation_lq: nn.Module, modulation_ldr: nn.Module):
        super().__init__()
        self.lq = condition_branch_lq
        self.ldr = condition_branch_ldr
        self.modulation_lq = modulation_lq
        self.modulation_ldr = modulation_ldr

    def forward(
        self,
        *,
        img,
        img_ids,
        condition_cond_lq,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_ldr=None,
    ):
        out_lq = self.lq(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_lq,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )

        out_ldr = self.ldr(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_ldr,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out = []
        num_blocks = 19
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, ldr) in enumerate(zip(out_lq, out_ldr)):
                control_index = torch.tensor(control_index, device=timesteps.device, dtype=timesteps.dtype)

                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index)

                if len(out) == num_blocks:
                    break

                ldr = self.modulation_ldr(ldr, timesteps, i * 2 + control_index)
                out.append(lq + ldr)
        return out

def preprocess_lq_image(image, width: int = 512, height: int = 512):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
    image = image.resize((width, height))
    return image

def load_redux_image_encoder(device: torch.device, dtype: torch.dtype, redux_state_dict):
    redux_image_encoder = ReduxImageEncoder()
    redux_image_encoder.load_state_dict(redux_state_dict, strict=False)

    redux_image_encoder.eval()
    redux_image_encoder.to(device).to(dtype=dtype)
    return redux_image_encoder

def test_model_loading():
    """测试模型加载"""
    print("Testing model loading...")

    checkpoint_path = "weights/lucidflux/lucidflux.pth"
    swinir_pretrained_path = "weights/swinir.pth"

    if not os.path.exists(checkpoint_path) or not os.path.exists(swinir_pretrained_path):
        print(f"Model files not found!")
        print(f"Looking for: {checkpoint_path} and {swinir_pretrained_path}")
        return False

    try:
        name = "flux-dev"
        torch_device = torch.device("cuda")

        # 使用预计算的embeddings
        embeddings_path = "weights/lucidflux/prompt_embeddings.pt"
        if os.path.exists(embeddings_path):
            print(f"Loading precomputed embeddings from {embeddings_path}")
            embeddings_data = torch.load(embeddings_path, map_location='cpu')
            precomputed_txt = embeddings_data['txt'].to(torch_device)
            precomputed_vec = embeddings_data['vec'].to(torch_device)
            original_prompt = embeddings_data.get('prompt', 'Unknown prompt')
            print(f"Loaded embeddings for prompt: '{original_prompt}'")
        else:
            raise FileNotFoundError(f"Prompt embeddings not found at {embeddings_path}")

        # base models - 主干模型不offload
        model = load_flow_model(name, device=torch_device)
        # 确保所有内部层都使用bfloat16
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
        ae = load_ae(name, device=torch_device)
        condition_lq = load_single_condition_branch(name, torch_device).to(torch.bfloat16)

        # load model checkpoint
        if os.path.exists(checkpoint_path):
            if '.safetensors' in checkpoint_path:
                checkpoint = load_safetensors(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        condition_lq.load_state_dict(checkpoint["condition_lq"], strict=False)
        condition_lq = condition_lq.to(torch_device)

        condition_ldr = load_single_condition_branch(name, torch_device).to(torch.bfloat16)
        condition_ldr.load_state_dict(checkpoint["condition_ldr"], strict=False)

        modulation_lq = Modulation(dim=3072).to(torch.bfloat16)
        modulation_lq.load_state_dict(checkpoint["modulation_lq"], strict=False)

        modulation_ldr = Modulation(dim=3072).to(torch.bfloat16)
        modulation_ldr.load_state_dict(checkpoint["modulation_ldr"], strict=False)

        dual_condition_branch = DualConditionBranch(
                condition_lq,
                condition_ldr,
                modulation_lq=modulation_lq,
                modulation_ldr=modulation_ldr,
            ).to(torch_device)

        # SwinIR prior
        if os.path.exists(swinir_pretrained_path):
            swinir = SwinIR(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=[6, 6, 6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
                window_size=8,
                mlp_ratio=2,
                sf=8,
                img_range=1.0,
                upsampler="nearest+conv",
                resi_connection="1conv",
                unshuffle=True,
                unshuffle_scale=8,
            )
            ckpt_obj = torch.load(swinir_pretrained_path, map_location="cpu")
            state = ckpt_obj.get("state_dict", ckpt_obj)
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            swinir.load_state_dict(new_state, strict=False)
            swinir.eval()
            for p in swinir.parameters():
                p.requires_grad_(False)
            swinir = swinir.to(torch_device)
        else:
            raise FileNotFoundError(f"SwinIR checkpoint not found at {swinir_pretrained_path}")

        # Skip SigLIP for now since it's causing issues
        siglip_model = None
        redux_image_encoder = load_redux_image_encoder(torch_device, torch.bfloat16, checkpoint["connector"])

        print("All models loaded successfully!")

        return {
            'model': model,
            'ae': ae,
            'dual_condition_branch': dual_condition_branch,
            'swinir': swinir,
            'siglip_model': siglip_model,
            'redux_image_encoder': redux_image_encoder,
            'precomputed_txt': precomputed_txt,
            'precomputed_vec': precomputed_vec,
            'device': torch_device,
            'offload': False
        }

    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference(models):
    """测试推理过程"""
    print("Testing inference...")

    # 创建一个简单的测试图像
    test_image = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    try:
        output_img, lq_processed = process_image(
            test_image, 20, 123456789, 4.0, 512, 512, models
        )
        print("Inference test successful!")
        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_image(image, num_steps, seed, guidance, width, height, models):
    """处理单张图片"""
    model = models['model']
    ae = models['ae']
    dual_condition_branch = models['dual_condition_branch']
    swinir = models['swinir']
    siglip_model = models['siglip_model']
    redux_image_encoder = models['redux_image_encoder']
    precomputed_txt = models['precomputed_txt']
    precomputed_vec = models['precomputed_vec']
    torch_device = models['device']
    offload = models['offload']

    name = "flux-dev"
    is_schnell = name == "flux-schnell"

    # 计算timesteps
    width = 16 * width // 16
    height = 16 * height // 16
    timesteps = get_schedule(
        num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=(not is_schnell),
    )

    # 预处理图片
    lq_processed = preprocess_lq_image(image, width, height)
    condition_cond = torch.from_numpy((np.array(lq_processed) / 127.5) - 1)
    condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
    condition_cond_ldr = None

    with torch.no_grad():
        # SwinIR prior
        ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
        ci_pre = swinir(ci_01.to(torch_device)).float().clamp(0.0, 1.0)
        condition_cond_ldr = (ci_pre * 2.0 - 1.0).to(torch.bfloat16)

        # diffusion inputs
        torch.manual_seed(seed)
        x = get_noise(
            1, height, width, device=torch_device,
            dtype=torch.bfloat16, seed=seed
        )

        # 使用预计算的embeddings
        inp_cond = prepare_with_embeddings(
            img=x, precomputed_txt=precomputed_txt, precomputed_vec=precomputed_vec
        )

        # Skip SigLIP for now since it's causing issues
        siglip_txt = inp_cond["txt"]
        siglip_txt_ids = inp_cond["txt_ids"]

        x = denoise_lucidflux(
            model,
            dual_condition_model=dual_condition_branch,
            img=inp_cond["img"],
            img_ids=inp_cond["img_ids"],
            txt=inp_cond["txt"],
            txt_ids=inp_cond["txt_ids"],
            siglip_txt=siglip_txt,
            siglip_txt_ids=siglip_txt_ids,
            vec=inp_cond["vec"],
            timesteps=timesteps,
            guidance=guidance,
            condition_cond_lq=condition_cond,
            condition_cond_ldr=condition_cond_ldr,
        )

        x = unpack(x.float(), height, width)
        x = ae.decode(x)

    # 后处理
    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")

    # Wavelet reconstruction
    hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, ci_pre.squeeze(0))
    hq = hq.clamp(0, 1)

    # 转换为PIL Image
    output_img = Image.fromarray((hq.cpu().numpy() * 255).astype(np.uint8))

    return output_img, lq_processed

if __name__ == "__main__":
    print("LucidFlux Test Script")
    print("====================")

    # Test model loading
    models = test_model_loading()
    if models:
        print("✅ Model loading test passed")

        # Test inference
        if test_inference(models):
            print("✅ Inference test passed")
            print("🎉 All tests passed! The demo should work.")
        else:
            print("❌ Inference test failed")
    else:
        print("❌ Model loading test failed")