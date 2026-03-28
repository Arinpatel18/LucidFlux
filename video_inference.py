import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
import subprocess

from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack
from src.flux.util import load_ae, load_flow_model
from src.flux.align_color import wavelet_reconstruction
from src.flux.lucidflux import (
    load_dual_condition_branch,
    load_lucidflux_weights,
    load_precomputed_embeddings,
    load_redux_image_encoder,
    load_siglip_model,
    load_swinir,
    move_modules_to_device,
    prepare_with_embeddings,
    preprocess_lq_image,
)
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="./video_results/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload", action='store_true')
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4)
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--swinir_pretrained", type=str, default=None)
    parser.add_argument("--siglip_ckpt", type=str, default="siglip2-so400m-patch16-512")
    return parser

def warp_previous(prev_img, current_img):
    # Convert numpy arrays from RGB to Grayscale for flow calculation
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = prev_img.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h)[:, np.newaxis], (1, w)).astype(np.float32)
    
    map_x += flow[..., 0]
    map_y += flow[..., 1]
    
    warped_prev = cv2.remap(prev_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped_prev

def process_video(args):
    torch_device = torch.device(args.device)
    name = "flux-dev"
    offload = args.offload
    is_schnell = name == "flux-schnell"

    os.makedirs(args.output_folder, exist_ok=True)

    # Load models
    embeddings_path = "weights/lucidflux/prompt_embeddings.pt"
    embeddings_data = load_precomputed_embeddings(embeddings_path, torch_device)
    precomputed_txt = embeddings_data["txt"]
    precomputed_vec = embeddings_data["vec"]

    model = load_flow_model(name, device=torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    lucidflux_weights = load_lucidflux_weights(args.checkpoint)
    dual_condition_branch = load_dual_condition_branch(
        name=name, lucidflux_weights=lucidflux_weights, device=torch_device,
        offload=offload, branch_dtype=torch.bfloat16,
    )

    swinir = load_swinir(torch_device, args.swinir_pretrained, offload)

    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    siglip_model = load_siglip_model(args.siglip_ckpt, torch_device, dtype, offload)
    redux_image_encoder = load_redux_image_encoder("cpu" if offload else torch_device, dtype, lucidflux_weights["connector"])

    width = 16 * args.width // 16
    height = 16 * args.height // 16
    timesteps = get_schedule(args.num_steps, (width // 8) * (height // 8) // (16 * 16), shift=(not is_schnell))

    # Read frames
    exts = (".png", ".jpg", ".jpeg")
    input_paths = sorted([
        os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
        if f.lower().endswith(exts)
    ])
    
    if len(input_paths) == 0:
        print("No frames found!")
        return

    prev_output = None
    prev_np_img = None

    for i, img_path in enumerate(input_paths):
        filename = f"frame_{i:05d}"
        
        # Load via cv2 and apply original preprocessing
        cv_img = cv2.imread(img_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # We need PIL for preprocess_lq_image compatibility
        lq_processed = preprocess_lq_image(img_path, args.width, args.height)
        current_frame = torch.from_numpy((np.array(lq_processed) / 127.5) - 1).float()

        if prev_output is not None:
             # Optical Flow warping on numpy representations
             curr_np = np.array(lq_processed)
             warped_np = warp_previous(prev_np_img, curr_np)
             
             # Convert warped numpy back to tensor [-1, 1]
             warped_tensor = torch.from_numpy((warped_np / 127.5) - 1).float()
             
             # Temporal consistency blending
             current_frame = 0.7 * current_frame + 0.3 * warped_tensor
             
        # Normalize and prepare condition_cond for inference
        condition_cond = current_frame.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)
        
        with torch.no_grad():
            ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
            if offload: swinir.to(torch_device)
            ci_pre = swinir(ci_01.to(torch_device)).float().clamp(0.0, 1.0)
            if offload: swinir.to("cpu")
            condition_cond_pre = (ci_pre * 2.0 - 1.0).to(torch.bfloat16)

            torch.manual_seed(args.seed)
            x = get_noise(1, height, width, device=torch_device, dtype=torch.bfloat16, seed=args.seed)
            inp_cond = prepare_with_embeddings(img=x, precomputed_txt=precomputed_txt, precomputed_vec=precomputed_vec)

            siglip_size = getattr(getattr(siglip_model, "config", None), "image_size", 512)
            siglip_pixel_values_pre = siglip_from_unit_tensor(ci_pre, size=(siglip_size, siglip_size))
            inputs = {"pixel_values": siglip_pixel_values_pre.to(device=torch_device, dtype=dtype)}
            
            if offload: siglip_model.to(torch_device)
            siglip_image_pre_fts = siglip_model(**inputs).last_hidden_state.to(dtype=dtype)
            if offload:
                siglip_model.to("cpu")
                torch.cuda.empty_cache()
                
            enc_dtype = redux_image_encoder.redux_up.weight.dtype
            if offload: redux_image_encoder.to(torch_device)
            image_embeds = redux_image_encoder(siglip_image_pre_fts.to(device=torch_device, dtype=enc_dtype))["image_embeds"]
            if offload:
                redux_image_encoder.to("cpu")
                torch.cuda.empty_cache()

            txt = inp_cond["txt"].to(device=torch_device, dtype=torch.bfloat16)
            txt_ids = inp_cond["txt_ids"].to(device=torch_device, dtype=torch.bfloat16)
            siglip_txt = torch.cat([txt, image_embeds.to(dtype=torch.bfloat16)], dim=1)
            B, L, C = txt_ids.shape
            extra_ids = torch.zeros((B, 1024, C), device=txt_ids.device, dtype=torch.bfloat16)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(dtype=torch.bfloat16)

            if offload:
                move_modules_to_device(torch_device, model, dual_condition_branch)
                torch.cuda.empty_cache()

            x = denoise_lucidflux(
                model, dual_condition_model=dual_condition_branch,
                img=inp_cond["img"], img_ids=inp_cond["img_ids"],
                txt=txt, txt_ids=txt_ids, siglip_txt=siglip_txt,
                siglip_txt_ids=siglip_txt_ids, vec=inp_cond["vec"],
                timesteps=timesteps, guidance=args.guidance,
                condition_cond_lq=condition_cond, condition_cond_pre=condition_cond_pre,
            )
            if offload:
                move_modules_to_device("cpu", model, dual_condition_branch)
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            x = unpack(x.float(), height, width)
            x = ae.decode(x)
            if offload:
                ae.decoder.cpu()
                torch.cuda.empty_cache()

        x1 = x.clamp(-1, 1)
        x1_hwc = rearrange(x1[-1], "c h w -> h w c")
        hq = wavelet_reconstruction((x1_hwc.permute(2, 0, 1) + 1.0) / 2, ci_pre.squeeze(0)).clamp(0, 1)
        
        # Save output frame
        out_path = os.path.join(args.output_folder, f"{filename}.png")
        hq_np_save = (hq.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(hq_np_save).save(out_path)
        print(f"[INFO] Processed {filename} -> {out_path}")
        
        # Store previous output for temporal consistency
        prev_output = hq.detach()
        # Create a numpy representation for optical flow calculation next iteration
        hq_np = (prev_output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        prev_np_img = hq_np

    # Reconstruct video using ffmpeg
    print("[INFO] Reconstructing video with ffmpeg...")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30",
            "-i", os.path.join(args.output_folder, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            os.path.join(args.output_folder, "final_output.mp4")
        ], check=True)
        print("[INFO] Video reconstruction complete. Saved as final_output.mp4")
    except Exception as e:
        print(f"[ERROR] Failed to reconstruct video: {e}")

if __name__ == "__main__":
    args = create_argparser().parse_args()
    process_video(args)
