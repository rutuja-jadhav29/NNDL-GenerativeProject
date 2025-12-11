import torch
import clip
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load model & preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load sample data
data_path = Path("data/flickr30k/cleaned_captions.csv")
img_dir = Path("data/flickr30k/cleaned_images")

df = pd.read_csv(data_path).sample(5, random_state=42)  # test on 5 random pairs

for _, row in df.iterrows():
    image_path = img_dir / row["image"]
    caption = row["caption"]

    # Load image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)

    # Compute similarity
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.cosine_similarity(image_features, text_features).item()

    # Show result
    plt.imshow(Image.open(image_path))
    plt.title(f"Caption: {caption}\nSimilarity: {similarity:.3f}")
    plt.axis("off")
    plt.show()
