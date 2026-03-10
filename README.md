<div align="center">
<h1>🎨 LucidFlux:<br/>Caption-Free Photo-Realistic Image Restoration via a Large-Scale Diffusion Transformer</h1>

### [**🌐 Website**](https://w2genai-lab.github.io/LucidFlux/) | [**📘 Arxiv**](http://arxiv.org/abs/2509.22414) | [**📄 Technical Report**](Technical_Report.pdf) | [**🤗 Models**](https://huggingface.co/W2GenAI/LucidFlux) | [**🔧 Fal-AI Demo&API**](https://fal.ai/models/fal-ai/lucidflux/playground) 
</div>

---
<img alt="abs_image" src="images/framework/abs_image.png" />

---
## 📰 News & Updates  

**[2025.03.10]** - We released the metadata for the clean images used in LucidFlux at [LucidFlux-Training-Data](https://huggingface.co/W2GenAI/LucidFlux/blob/main/LucidFlux-Training-Data.tar.gz) and the filtering pipeline in ```tools/filtering_pipeline.py```.

**[2025.02.06]** - LucidFlux is accepted by **ICLR'26**.

**[2025.10.07]** — Thanks to [smthemex](https://github.com/smthemex) for developing [ComfyUI_LucidFlux](https://github.com/smthemex/ComfyUI_LucidFlux), which enables LucidFlux to run with **as little as 8 GB–12 GB of memory** through the ComfyUI integration. 

**[2025.10.06]**  -- LucidFlux now supports **offload** and **precomputed prompt embeddings**, eliminating the need to load T5 or CLIP during inference. These improvements reduce memory usage significantly — inference can now run with **as little as 28 GB VRAM**, greatly enhancing deployment efficiency.  

**[2025.10.05]**  -- LucidFlux has been officially added to the **Fal AI Playground**! You can now try the **online demo** and access the **Fal API** directly here:  
👉 [**LucidFlux on Fal AI**](https://fal.ai/models/fal-ai/lucidflux/playground)  

---

Let us know if this works!

## 👥 Authors

> [**Song Fei**](https://feisong123.github.io)<sup>1</sup>\*, [**Tian Ye**](https://owen718.github.io/)<sup>1</sup>\*‡, [**Lujia Wang**](https://scholar.google.com/citations?user=c2_syKsAAAAJ)<sup>1</sup> , [**Lei Zhu**](https://sites.google.com/site/indexlzhu/home)<sup>1,2</sup>†
>
> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)  
> <sup>2</sup>The Hong Kong University of Science and Technology  
>
> \*Equal Contribution, ‡Project Leader, †Corresponding Author

---

## 🌟 What is LucidFlux?

LucidFlux is a caption-free universal image restoration framework that leverages a lightweight dual-branch conditioner and adaptive modulation to guide a large diffusion transformer (Flux.1) with minimal overhead, achieving robust, high-fidelity restoration without relying on text prompts or MLLM captions.

## 📊 Performance Benchmarks

<div align="center">

### 📈 Quantitative Results

<img alt="quantitative_comparison" src="images/framework/quantitative_comparison.png" />
<img alt="quantitative_comparison_commercial" src="images/framework/quantitative_comparison_commercial.png" />

</div>

---

## 🎭 Gallery & Examples

<div align="center">

### 🎨 LucidFlux Gallery

---

### 🔍 Comparison with Open-Source Methods

<table>
<tr align="center">
    <td width="200"><b>LQ</b></td>
    <td width="200"><b>SinSR</b></td>
    <td width="200"><b>SeeSR</b></td>
    <td width="200"><b>SUPIR</b></td>
    <td width="200"><b>DreamClear</b></td>
    <td width="200"><b>Ours</b></td>
</tr>
<tr align="center"><td colspan="6"><img src="images/comparison/040.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/041.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/111.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/123.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/160.jpg" width="1200"></td></tr>
</table>

<details>
<summary>Show more examples</summary>

<table>
<tr align="center"><td colspan="6"><img src="images/comparison/013.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/079.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/082.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/137.jpg" width="1200"></td></tr>
<tr align="center"><td colspan="6"><img src="images/comparison/166.jpg" width="1200"></td></tr>
</table>

</details>

---

### 💼 Comparison with Commercial Models

<table>
<tr align="center">
    <td width="200"><b>LQ</b></td>
    <td width="200"><b>HYPIR-FLUX</b></td>
    <td width="200"><b>Topaz</b></td>
    <td width="200"><b>Seedream 4.0</b></td>
    <td width="200"><b>MeiTu SR</b></td>
    <td width="200"><b>Gemini-NanoBanana</b></td>
    <td width="200"><b>Ours</b></td>
</tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_061.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_094.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_205.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_209.jpg" width="1400"></td></tr>
</table>

<details>
<summary>Show more examples</summary>

<table>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_062.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_160.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_111.jpg" width="1400"></td></tr>
<tr align="center"><td colspan="7"><img src="images/commercial_comparison/commercial_123.jpg" width="1400"></td></tr>
</table>

</details>
</div>

---

## 🏗️ Model Architecture

<div align="center">
<img src="images/framework/framework.png" alt="LucidFlux Framework Overview" width="1200"/>
<br>
<em><strong>Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer</strong></em>
</div>

Our unified framework consists of **four critical components in the training workflow**:


**🎨 Dual-Branch Conditioner for Low-Quality Image Conditioning**

**🎯 Timestep and Layer-Adaptive Condition Injection**

**🔄 Semantic Priors from Siglip for Caption-Free Semantic Alignment**

**🔤 Scaling Up Real-world High-Quality Data for Universal Image Restoration**

## 🚀 Quick Start

> ⚠️ The default setup requires roughly 28 GB of GPU VRAM. 

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/W2GenAI-Lab/LucidFlux.git
cd LucidFlux

# Create conda environment
conda create -n lucidflux python=3.11
conda activate lucidflux

# Install PyTorch (CUDA 12.8 wheels)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
pip install --upgrade timm

```

### Inference

Prepare models in 2 steps, then run a single command.

1) Login to Hugging Face (required for gated FLUX.1-dev). Skip if already logged-in.

```bash
python -m tools.hf_login --token "$HF_TOKEN"
```

2) Download required weights to fixed paths and export env vars

```bash
# FLUX.1-dev (flow+ae), SwinIR prior, T5, CLIP, SigLIP and LucidFlux checkpoint to ./weights
python -m tools.download_weights --dest weights

# Exports FLUX_DEV_FLOW/FLUX_DEV_AE to your shell (Linux/macOS)
source weights/env.sh

# Windows: open `weights\env.sh`, replace each leading `export` with `set`, then paste those commands into Command Prompt
```


Run inference (uses fixed relative paths):

```bash
bash inference.sh
```

> ℹ️ LucidFlux builds on Flux-based generative priors. Restored images can differ from the low-quality input because the model removes degradations and hallucinates realistic details by design. Visual discrepancies are expected and indicate the generative nature of the method.

You can also obtain results of LucidFlux on RealSR and RealLQ250 from Hugging Face: [**LucidFlux**](https://huggingface.co/W2GenAI/LucidFlux).

### Data and Filtering Pipeline

We have released the metadata for the clean images used in LucidFlux at [LucidFlux-Training-Data](https://huggingface.co/W2GenAI/LucidFlux/blob/main/LucidFlux-Training-Data.tar.gz), and the filtering pipeline in ```tools/filtering_pipeline.py```. Images can be downloaded from the provided `image_url`, and filtering scores can be computed with:

```bash
# download the image
python tools/download_from_image_url.py "image_url in metadata"

# compute filtering scores
python tools/filtering_pipeline.py /path/to/image.jpg
```

The LSDIR dataset can be downloaded from [LSDIR](https://huggingface.co/ofsoundof/LSDIR), and degradation synthesis follows the pipeline from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). If more data is needed, please also refer to our [MultiAspect-4K-1M](https://github.com/W2GenAI-Lab/UltraFlux) dataset.

---

## 🚀 Updates
For the purpose of fostering research and the open-source community, we plan to open-source the entire project, encompassing training, inference, weights, etc. Thank you for your patience and support! 🌟
- [x] Release github repo.
- [x] Release inference code.
- [x] Release model checkpoints.
- [x] Release arXiv paper.
- [ ] Release training code.
- [x] Release the training data and filtering pipeline.




## 📝 Citation

If you find LucidFlux useful for your research, please cite our report:

```bibtex
@article{fei2025lucidflux,
  title={LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer},
  author={Fei, Song and Ye, Tian and Wang, Lujia and Zhu, Lei},
  journal={arXiv preprint arXiv:2509.22414},
  year={2025}
}
```
---

## 🪪 License

The provided code and pre-trained weights are licensed under the [FLUX.1 \[dev\]](LICENSE).

## 🙏 Acknowledgments

- This code is based on [FLUX](https://github.com/black-forest-labs/flux). Some code are brought from [DreamClear](https://github.com/shallowdream204/DreamClear), [x-flux](https://github.com/XLabs-AI/x-flux). We thank the authors for their awesome work.

- 🏛️ Thanks to our affiliated institutions for their support.
- 🤝 Special thanks to the open-source community for inspiration.

---

## 📬 Contact

For any questions or inquiries, please reach out to us:

- **Song Fei**: `sfei285@connect.hkust-gz.edu.cn`
- **Tian Ye**: `tye610@connect.hkust-gz.edu.cn`

## 🧑‍🤝‍🧑 WeChat Group
<details>
  <summary>点击展开二维码（WeChat Group QR Code）</summary>

  <br>

  <img src="images/framework/QR-code.jpg"
       alt="WeChat Group QR"
       width="320">

</details>

<details>
  <summary>如果群二维码过期，点击展开作者微信二维码（Author WeChat QR Code）</summary>

  <br>

  <img src="https://github.com/user-attachments/assets/8d4e11eb-119c-4fe7-a901-d112b051eaba"
       alt="WeChat Group QR"
       width="320">

</details>


</div>
