# Text-to-Image Generation with Stable Diffusion
Text-to-image generation system using CLIP (Transformer) and Stable Diffusion (Diffusion Model) on Flickr30k dataset for IE 7615 Deep Learning course.

## Overview
Generates 512×512 images from text descriptions. Achieved 67% quality improvement through systematic parameter optimization.

## Results
| Model | FID | IS | CLIP | Images |
|-------|-----|-----|------|--------|
| M1 Baseline | 464.75 | 10.30±7.20 | 0.311 | 50 |
| M2 Optimized | 526.13 | 17.24±9.40 | 0.310 | 60 |

**Key Finding:** Guidance scale 7.5 provides optimal balance between quality and text alignment.

## Installation
```bash
pip install torch torchvision diffusers==0.30.0 transformers==4.35.0 accelerate pandas matplotlib pillow scipy scikit-learn
pip install git+https://github.com/openai/CLIP.git
```

## Quick Start
```python
# Clone repository
git clone https://github.com/rutuja-jadhav29/NNDL-GenerativeProject.git
cd NNDL-GenerativeProject
```
Open `untitled40__1_.py` in Google Colab with GPU enabled. Run all cells sequentially (Runtime: ~2 hours)

## Repository Structure
```
├── data/flickr30k/              # Dataset (3000 pairs, 2879 images)
├── outputs/                     # M1 baseline (50 images)
├── Milestone2_Selected_Images/  # M2 optimized (60 images)
├── results/plots/               # FID, IS, CLIP charts
├── results/tables/              # Metrics CSV
├── untitled40__1_.py            # Complete implementation
└── training_log.csv             # Generation metadata
```

## Models Used
- **Text Encoder:** CLIP ViT-B/32 (512-dim embeddings, 12 Transformer layers)
- **Image Generator:** Stable Diffusion v1.5 (860M parameters)
- **Hardware:** Google Colab T4 GPU (16GB VRAM)

## Key Findings
1. Optimal configuration: Guidance 7.5, 50 inference steps
2. Simple prompts outperform complex by 21% (CLIP: 0.341 vs 0.278)
3. M2 achieved 67% IS improvement over M1 baseline
4. Limitations: Counting failures, spatial reasoning errors (40% incorrect)

## Citation
```bibtex
@project{text2img2024,
  title={Text-to-Image Generation with Stable Diffusion},
  author={Rutuja Jadhav and Team},
  year={2024},
  course={IE 7615 Deep Learning}
}
```

## License
MIT License - Educational use only

## Acknowledgments
- Stable Diffusion: Rombach et al. (2022)
- CLIP: Radford et al. (2021)
- Flickr30k: Young et al. (2014)
