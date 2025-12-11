# NNDL-GenerativeProject

## Text-to-Image Generation Using Stable Diffusion

A comprehensive implementation and evaluation of text-to-image generation using Stable Diffusion v1.5 and CLIP text encoders. This project demonstrates the complete pipeline from dataset preparation through model evaluation with multiple quantitative metrics.


## Project Structure
```
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── flickr30k/
│       ├── cleaned_images/
│       └── cleaned_captions.csv
│
├── milestone1/
│   ├── outputs/
│   ├── baseline_text_to_embedding.ipynb
│   ├── E7615_Project2-Milestone1.pdf
│   ├── preprocess_flickr30k.py
│   └── validate_clip_alignment.py
│
├── milestone2/
│   ├── M2_Outputs/
│   ├── Generative_Project_Milestone2.ipynb
│   ├── Milestone_2_Generative_project.ipynb
│   └── training_log.csv
│
├── milestone3/
│   ├── Art_Styles.ipynb
│   ├── Generative_Project_Milestone3.ipynb
│   └── Milestone3Final.ipynb
│
├──
│   ├── NNDL_FinalPPt.pptx
│   └── Report_NNDL.pdf
│
└── .git/


```

## Installation

### Prerequisites

- Python 3.8 or higher

- CUDA-capable GPU (recommended for training/generation)

- 16GB+ RAM

10GB+ free disk space

## Setup

### 1. Clone the repository and 
```
git clone https://github.com/rutuja-jadhav29/NNDL-GenerativeProject.git
   cd NNDL-GenerativeProject
```

2. Create virtual environment
```
python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
 ```

3. Install dependencies
```
pip install -r requirements.txt
```

## Dataset

### Flickr30k Dataset

We use the Flickr30k dataset containing 31,000 images with five captions each.
Download Instructions:

Request access to Flickr30k from the official source
Download and extract to data/flickr30k_images/
Run preprocessing script



Dataset Statistics

Total Images: 31,000

Cleaned Images: 2,879 (after removing corrupted files)

Captions per Image: 5

Image Size: 224×224 pixels

Dataset Size: ~8GB (original), ~500MB (cleaned)


## Evaluation Metrics
### Fréchet Inception Distance (FID)

Measures distribution similarity between real and generated images using Inception-V3 features.

Lower is better

Baseline: 404.25

Stable Diffusion: 362.79 ✅

### Inception Score (IS)

Evaluates image quality and diversity based on classifier predictions.

Higher is better

Baseline: 1.0

Stable Diffusion: 1.3 ✅


## Key Achievements

✅ 10.3% FID Improvement: Stable Diffusion achieves FID of 362.79 vs 404.25 baseline

✅ 30% IS Improvement: Inception Score increased from 1.0 to 1.3

✅ Optimal Parameters Identified: Guidance scale 7.5 and 50 inference steps

✅ Comprehensive Evaluation: FID, IS, and CLIP similarity metrics

✅ Parameter Sensitivity Analysis: Guidance scale and embedding architecture comparison

## Features

🖼️ Text-to-Image Generation: Generate images from natural language prompts

📊 Multiple Evaluation Metrics: FID, Inception Score, CLIP similarity

🔧 Parameter Tuning: Guidance scale and inference steps optimization

📈 Comprehensive Logging: Training logs and result tracking

🎨 Qualitative Analysis: Side-by-side visual comparisons

🔬 Embedding Sensitivity: CLIP ViT-B/32 vs ViT-B/16 comparison
