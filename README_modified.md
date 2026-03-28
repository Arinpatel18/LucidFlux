# LucidFlux: HPC / Cluster Optimized Version

This version of the codebase has been extensively customized and optimized to run on HPC clusters using SLURM job scheduling, specifically dealing with limited GPU VRAM (e.g., 16-40GB GPUs) and strictly separated storage partitions (HOME vs. SCRATCH) to avoid disk quota issues.

## 🚀 Key Features & Modifications

1. **Storage Optimization (HOME vs SCRATCH):** 
   - Codebase resides in the limited `HOME` directory.
   - All bulky files (Base Weights, HuggingFace Caches, Datasets, and Checkpoint Outputs) are safely redirected to the `SCRATCH` partition to bypass strict storage quotas.
   
2. **DeepSpeed VRAM Optimization:**
   - Modified the DeepSpeed config (`ds_zero2.yaml`) to use **CPU Offloading** (`offload_optimizer_device: cpu`). This allows training a massive 12B parameter model on a single GPU by storing the optimizer states on the system RAM instead of GPU VRAM.

3. **AdamW Native Support:**
   - Refactored `train.py` to support `adamw` natively. The original codebase strictly forced `adafactor`, which inherently crashes DeepSpeed's CPU offloader. 
   
4. **SLURM Automation:**
   - Added robust SLURM `.sh` scripts (`train_dit.sh` and `inference_job.sh`) to fully automate node allocation, Conda initialization, and environment variable pathing dynamically.

---

## 🛠️ Environment Setup

### 1. Storage Configuration
All heavy assets must be stored in the scratch space for safety. Ensure the following directory exists:
`/scratch/data/{username}/LucidFlux`

### 2. Symlinking Weights
To avoid changing hardcoded paths in the source code, keep weights in the scratch drive but symbolically link them back into the code directory:
```bash
# In your main project folder:
ln -s /scratch/data/{username}/LucidFlux/weights weights
```

### 3. Caching and Environment Variables
Due to missing baseline `env.sh` scripts, HuggingFace variables and model paths are automatically injected via the SLURM job scripts directly. They point cache and temp setups precisely to `/scratch/data/.../huggingface`.

---

## 🏋️‍♂️ Training the Model

1. **Check the Configuration:** Edit `train_configs/train_LucidFlux.yaml` if you need to alter learning rates, image sizes, or epochs. Currently, it defaults to:
   - `train_batch_size: 1`
   - `optimizer_type: adamw`
   - `report_to: tensorboard`

2. **Submit Training Job:**
   Run the following SLURM command:
   ```bash
   sbatch train_dit.sh
   ```
   *Logs can be viewed dynamically via:* `tail -f /scratch/data/{username}/LucidFlux/train_<jobid>.out`

3. **Where are the Trained Weights Saved?**
   During training, model weights and checkpoint states are saved directly to the scratch drive to prevent HOME quota overflows:
   📂 **Output Path:** `/scratch/data/{username}/LucidFlux/output/`
   Inside you will find `checkpoint-<steps>/lucidflux.pth`. 
   
   *(You can copy your desired `.pth` back to your main folder for inference).*

---

## 🎨 Running Inference

1. **Check Weights Configuration:**
   Place the specific version of the trained model you want to test in the project folder, or alter `inference.sh` to point to it (e.g., `--checkpoint lucidflux_trained.pth`).

2. **Submit the Inference Job:**
   Because loading the graph requires extensive system RAM, always use the SLURM inference job:
   ```bash
   sbatch inference_job.sh
   ```

3. **Where are the Generated Images Saved?**
   Once the job log outputs `===== INFERENCE DONE =====`, your generated images are saved seamlessly to your Scratch drive block:
   📂 **Output Path:** `/scratch/data/{username}/LucidFlux/outputs-trained/`

   You can then copy these images back to your desktop or HOME workspace to visualize them.

---

## 🎥 Running Video Inference

We have extended the base LucidFlux pipeline to support temporally consistent video upscaling and enhancement. The video inference pipeline uses Optical Flow (Farneback) and temporal blending to maintain consistency between frames without requiring any structural model retuning.

### 1. Prerequisites for Video Processing
Ensure you install the required packages to handle frame reading, optical flow, and video compilation:
```bash
pip install opencv-python
conda install -c conda-forge ffmpeg
```

### 2. Prepare the Input
Extract your input video into individual frames (e.g., `frame_00001.png`, `frame_00002.png`) and place them into a folder.

### 3. Submit the Video Inference Job
Edit the input path in `video_inference_job.sh` under `--input_folder` to point to your frames directory.

Then submit the SLURM job:
```bash
sbatch video_inference_job.sh
```

### 4. Video Reassembly
The `video_inference.py` script automatically processes the frames sequentially, applies temporal smoothing using the previous frame and optical flow, saves individual enhanced frames, and finally reconstructs a cohesive `.mp4` video utilizing `ffmpeg`. The final outputs are stored in the folder defined by `--output_folder`.