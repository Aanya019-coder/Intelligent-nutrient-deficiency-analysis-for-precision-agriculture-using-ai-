import kagglehub
import os
import shutil

# Download datasets
print("Downloading PlantVillage (emmarex/plantdisease)...")
path1 = kagglehub.dataset_download("emmarex/plantdisease")
print(f"PlantVillage downloaded to: {path1}")

print("\nDownloading Rice Nutrient Deficiency (guy007/nutrientdeficiencysymptomsinrice)...")
path2 = kagglehub.dataset_download("guy007/nutrientdeficiencysymptomsinrice")
print(f"Rice Nutrient downloaded to: {path2}")

# Paths for the final organized dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)

print(f"\nDatasets are ready at {path1} and {path2}")
print("Please manually organize them into the folder structure expected by train.py:")
print("dataset/")
print("  train/")
print("    Nitrogen_Deficiency/")
print("    Phosphorus_Deficiency/")
print("    Potassium_Deficiency/")
print("    Iron_Deficiency/")
print("    Zinc_Deficiency/")
print("    Magnesium_Deficiency/")
print("    Healthy/")
print("  val/")
print("  test/")
