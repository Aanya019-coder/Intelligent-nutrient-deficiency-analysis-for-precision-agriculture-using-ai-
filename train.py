"""
TRAINING SCRIPT — Run this on Google Colab (GPU recommended)
=============================================================
This trains the EfficientNet-B3 + SE + CBAM model exactly as described
in the research paper. After training, download best_model.weights.h5
and place it in the nutrient_app/model/ folder.

Google Colab Setup:
  !pip install tensorflow keras matplotlib pillow
  Upload this file and run all cells.

Dataset (download from Kaggle):
  https://www.kaggle.com/datasets/emmarex/plantdisease
  https://www.kaggle.com/datasets/guy007/nutrientdeficiencysymptomsinrice

Folder structure expected:
  dataset/
    train/
      Nitrogen_Deficiency/  (or N_Def, etc.)
      Phosphorus_Deficiency/
      Potassium_Deficiency/
      Iron_Deficiency/
      Zinc_Deficiency/
      Magnesium_Deficiency/
      Healthy/
    val/   (same structure)
    test/  (same structure)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 1e-4
NUM_CLASSES = 7
DATA_DIR    = './dataset'   # <-- change to your dataset path
MODEL_DIR   = './model'
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ['Iron_Deficiency', 'Magnesium_Deficiency', 'Nitrogen_Deficiency',
               'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Zinc_Deficiency', 'Healthy']

# ── SE Block ──────────────────────────────────────────────────────
def se_block(x, ratio=16):
    channels = x.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Dense(max(channels // ratio, 1), activation='relu')(se)
    se = keras.layers.Dense(channels, activation='sigmoid')(se)
    se = keras.layers.Reshape((1, 1, channels))(se)
    return keras.layers.Multiply()([x, se])

# ── CBAM Block ────────────────────────────────────────────────────
def cbam_block(x):
    channels = x.shape[-1]
    # Channel attention
    avg_p = keras.layers.GlobalAveragePooling2D()(x)
    max_p = keras.layers.GlobalMaxPooling2D()(x)
    avg_p = keras.layers.Dense(max(channels // 8, 1), activation='relu')(avg_p)
    avg_p = keras.layers.Dense(channels, activation='sigmoid')(avg_p)
    max_p = keras.layers.Dense(max(channels // 8, 1), activation='relu')(max_p)
    max_p = keras.layers.Dense(channels, activation='sigmoid')(max_p)
    ch_att = keras.layers.Add()([avg_p, max_p])
    ch_att = keras.layers.Reshape((1, 1, channels))(ch_att)
    x = keras.layers.Multiply()([x, ch_att])
    # Spatial attention
    avg_s = keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_s = keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    spatial = keras.layers.Concatenate(axis=-1)([avg_s, max_s])
    spatial = keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial)
    return keras.layers.Multiply()([x, spatial])

# ── Build Model ───────────────────────────────────────────────────
# ── Build Model ───────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = keras.applications.EfficientNetB3(
        include_top=False, weights='imagenet',
        input_tensor=inputs, pooling=None
    )
    base.trainable = False  # Phase 1: frozen backbone

    x = base.output
    x = se_block(x, ratio=16)
    x = cbam_block(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='NutriScan_EfficientNetB3_v2')
    return model

# ── Data Pipeline ─────────────────────────────────────────────────
def make_dataset(subset, augment=False):
    path = os.path.join(DATA_DIR, subset)
    if not os.path.exists(path):
        print(f"Directory {path} missing, creating empty placeholder...")
        os.makedirs(path, exist_ok=True)
        for cls in CLASS_NAMES: os.makedirs(os.path.join(path, cls), exist_ok=True)

    ds = keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(subset == 'train'),
        seed=42,
    )

    preprocess = keras.applications.efficientnet.preprocess_input

    if augment:
        augmentor = keras.Sequential([
            keras.layers.RandomFlip('horizontal_and_vertical'),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomBrightness(0.2),
            keras.layers.RandomContrast(0.2),
        ])
        ds = ds.map(lambda x, y: (augmentor(preprocess(x), training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (preprocess(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)

# ── Class weights (handle imbalance) ─────────────────────────────
def compute_class_weights(train_ds):
    counts = np.zeros(NUM_CLASSES)
    # Check if dataset is empty
    if len(train_ds) == 0: return {i: 1.0 for i in range(NUM_CLASSES)}
    for _, labels in train_ds:
        counts += labels.numpy().sum(axis=0)
    total = counts.sum()
    if total == 0: return {i: 1.0 for i in range(NUM_CLASSES)}
    weights = {i: total / (NUM_CLASSES * (counts[i] + 1e-6)) for i in range(NUM_CLASSES)}
    return weights

# ── Main Training ─────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  NutriScan AI — Best Performance Mode")
    print("  EfficientNet-B3 + SE + CBAM | IILM University")
    print("=" * 60)

    train_ds = make_dataset('train', augment=True)
    val_ds   = make_dataset('val',   augment=False)
    test_ds  = make_dataset('test',  augment=False)

    class_weights = compute_class_weights(train_ds)
    print(f"\nClass weights: {class_weights}")

    model = build_model()
    
    # ── PHASE 1: Warmup ─────────────
    print("\n[Phase 1] Head Warmup (Frozen Backbone)...")
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    cb_p1 = [
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.weights.h5'),
                                     monitor='val_accuracy', save_best_only=True,
                                     save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history_p1 = model.fit(train_ds, validation_data=val_ds, epochs=15, 
                          callbacks=cb_p1, class_weight=class_weights)

    # ── PHASE 2: Deep Fine-tuning ─────────────
    print("\n[Phase 2] Deep Fine-Tuning (Unfreezing Backbone)...")
    model.trainable = True
    # Still keep some early layers frozen for stability
    freeze_until = int(len(model.layers[1].layers) * 0.5) if isinstance(model.layers[1], keras.Model) else 100
    if len(model.layers) > 1 and hasattr(model.layers[1], 'layers'):
        for layer in model.layers[1].layers[:freeze_until]: layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LR * 0.1),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    cb_p2 = [
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.weights.h5'),
                                     monitor='val_accuracy', save_best_only=True,
                                     save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    history_p2 = model.fit(train_ds, validation_data=val_ds, epochs=35, 
                          callbacks=cb_p2, class_weight=class_weights)

    # ── Final Save & Curve generation ─────────────
    print("\nEvaluation & Save...")
    if os.path.exists(os.path.join(MODEL_DIR, 'best_model.weights.h5')):
        model.load_weights(os.path.join(MODEL_DIR, 'best_model.weights.h5'))
    
    # Save curves
    try:
        all_acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
        plt.figure(figsize=(10, 4))
        plt.plot(all_acc, label='Train Acc')
        plt.title('Training Best Accuracy Evolution')
        plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
    except: pass

    print(f"Training Complete. Best weights at {MODEL_DIR}/best_model.weights.h5")

if __name__ == '__main__':
    train()
