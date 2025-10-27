import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from pathlib import Path
import re, os

# --- Paths ---
ROOT = Path("data/flickr30k")
IMG_DIR = ROOT / "flickr30k_images"
CAP_FILE = ROOT / "results.csv"
OUT_IMG_DIR = ROOT / "cleaned_images"
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- Clean captions ---
def clean_caption(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

# --- Read the pipe-separated file ---
df = pd.read_csv(CAP_FILE, sep="|", engine="python")

# Remove whitespace from column names
df.columns = df.columns.str.strip()

# Now confirm columns
print("Columns found:", df.columns.tolist())

# --- Select only relevant columns ---
df = df[["image_name", "comment"]]
df.rename(columns={"image_name": "image", "comment": "caption"}, inplace=True)

# --- Clean and sample captions ---
df["caption"] = df["caption"].apply(clean_caption)
df = df.drop_duplicates(subset=["image", "caption"]).sample(n=3000, random_state=42)

# --- Resize and save images ---
resize = transforms.Compose([transforms.Resize((224, 224))])

for img_name in tqdm(df["image"].unique(), desc="Resizing images"):
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        print(f"⚠️ Skipping {img_name} — image file missing")
        continue
    try:
        im = Image.open(img_path).convert("RGB")
        im = resize(im)
        im.save(OUT_IMG_DIR / img_name)
    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")

# --- Save cleaned captions ---
out_path = ROOT / "cleaned_captions.csv"
df.to_csv(out_path, index=False)
print(f"✅ Done! Cleaned data saved to {out_path}")
