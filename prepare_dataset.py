import os
import shutil
import random

# Source paths from paths.txt
SOURCE1 = r"C:\Users\HP\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
SOURCE2 = r"C:\Users\HP\.cache\kagglehub\datasets\guy007\nutrientdeficiencysymptomsinrice\versions\1\rice_plant_lacks_nutrients"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

CLASS_MAP = {
    'Nitrogen_Deficiency':   [(SOURCE2, 'Nitrogen(N)')],
    'Phosphorus_Deficiency': [(SOURCE2, 'Phosphorus(P)')],
    'Potassium_Deficiency':  [(SOURCE2, 'Potassium(K)')],
    'Healthy':               [(SOURCE1, 'Potato___healthy'), (SOURCE1, 'Tomato_healthy'), (SOURCE1, 'Pepper__bell___healthy')],
    'Iron_Deficiency':       [], # Missing in these datasets
    'Magnesium_Deficiency':  [], # Missing in these datasets
    'Zinc_Deficiency':       [], # Missing in these datasets
}

def prepare():
    for subset in ['train', 'val', 'test']:
        for cls in CLASS_MAP.keys():
            os.makedirs(os.path.join(DATA_DIR, subset, cls), exist_ok=True)

    for target_cls, sources in CLASS_MAP.items():
        all_images = []
        for src_root, src_folder in sources:
            src_path = os.path.join(src_root, src_folder)
            if os.path.exists(src_path):
                imgs = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_images.extend(imgs)
        
        if not all_images:
            print(f"Skipping {target_cls} (no source images found)")
            continue

        random.shuffle(all_images)
        
        # Split 70/15/15
        n = len(all_images)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        splits = {
            'train': all_images[:train_end],
            'val':   all_images[train_end:val_end],
            'test':  all_images[val_end:]
        }

        for subset, imgs in splits.items():
            dest_dir = os.path.join(DATA_DIR, subset, target_cls)
            print(f"Copying {len(imgs)} images to {subset}/{target_cls}...")
            for img_path in imgs:
                shutil.copy(img_path, dest_dir)

if __name__ == "__main__":
    prepare()
