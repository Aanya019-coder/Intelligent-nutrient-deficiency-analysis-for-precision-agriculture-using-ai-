# NutriScan AI — Nutrient Deficiency Analysis System
### EfficientNet-B3 + SE + CBAM | IILM University B.Tech AI-ML 2025–26

**Authors:** Aarya Chaudhary  
**Supervisor:** Dr. Swati Vashisht | School of CSE, IILM University, Greater Noida

---

## What This Project Does

Upload a crop leaf image → get:
- 🔬 **Deficiency class** (N / P / K / Fe / Zn / Mg / Healthy) with confidence %
- 📊 **Probability chart** for all 7 classes
- 🔥 **Grad-CAM heatmap** showing which leaf regions the model focused on
- 🌱 **Crop-specific fertilizer recommendation** (fertilizer type, rate, method, urgency)
- 🌍 **SDG-2 alignment** note

---

## Quick Start (5 minutes)

### Step 1 — Install dependencies
```bash
pip install flask tensorflow numpy pillow matplotlib
```

### Step 2 — Run the app
```bash
cd nutrient_app
python app.py
```

### Step 3 — Open browser
```
http://127.0.0.1:5000
```

That's it! The app runs with ImageNet pretrained weights immediately.  
To use your trained weights, see the Training section below.

---

## Project Structure

```
nutrient_app/
├── app.py              ← Main Flask app (model + API + routes)
├── train.py            ← Training script (run on Google Colab)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── model/
│   └── best_model.weights.h5   ← Place trained weights here
└── templates/
    └── index.html      ← Web UI (upload, results, Grad-CAM, recommendations)
```

---

## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNet-B3 backbone (ImageNet pretrained)
    ↓
SE Block (Squeeze-and-Excitation — channel attention)
    ↓
CBAM Block (channel + spatial attention)
    ↓
Global Average Pooling
    ↓
Dense(512, relu) → Dropout(0.3)
    ↓
Softmax Output (7 classes)
```

**Performance (from paper):**
| Metric | Score |
|--------|-------|
| Accuracy | 93.8% |
| F1-Macro | 92.6% |
| Precision | 93.2% |
| Recall | 92.1% |
| Parameters | 12.3M |

---

## Training Your Own Model (Google Colab)

### Step 1 — Get datasets

Download from Kaggle:
- PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease
- Rice Nutrient: https://www.kaggle.com/datasets/guy007/nutrientdeficiencysymptomsinrice

### Step 2 — Organize dataset folder

```
dataset/
├── train/
│   ├── Nitrogen_Deficiency/    (images here)
│   ├── Phosphorus_Deficiency/
│   ├── Potassium_Deficiency/
│   ├── Iron_Deficiency/
│   ├── Zinc_Deficiency/
│   ├── Magnesium_Deficiency/
│   └── Healthy/
├── val/   (same structure, 15% of data)
└── test/  (same structure, 15% of data)
```

### Step 3 — Run training on Colab

```python
# In Google Colab:
!pip install tensorflow
!python train.py
```

Training takes ~2–3 hours on Colab GPU (T4/A100).

### Step 4 — Download and use weights

After training, download `model/best_model.weights.h5` from Colab.  
Place it in `nutrient_app/model/` and restart the app — it will auto-load.

---

## Supported Crops & Nutrients

| Crop | Nutrients Covered |
|------|------------------|
| 🌾 Rice | N, P, K, Fe, Zn, Mg, Healthy |
| 🌿 Wheat | N, P, K, Fe, Zn, Mg, Healthy |
| 🌽 Maize | N, P, K, Fe, Zn, Mg, Healthy |
| 🍅 Tomato | N, P, K, Fe, Zn, Mg, Healthy |

---

## References (from Research Paper)

[4] Mohanty et al. (2016) — PlantVillage deep learning benchmark  
[10] Tan & Le (2019) — EfficientNet compound scaling  
[11] Espejo-Garcia et al. (2022) — EfficientNet-B4 for nutrient deficiency (baseline we outperform)  
[12] Bera et al. (2024) — PND-Net (baseline we outperform)  
[14] Selvaraju et al. (2020) — Grad-CAM explainability  
[15] Hu et al. (2018) — Squeeze-and-Excitation networks  
[16] Woo et al. (2018) — CBAM attention module  
