import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ── CONFIG ────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_P1   = 20
EPOCHS_P2   = 80
LR_P1       = 1e-4
LR_P2       = 1e-5
NUM_CLASSES = 7
DATA_DIR    = './dataset'
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
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = keras.applications.EfficientNetB3(
        include_top=False, weights='imagenet',
        input_tensor=inputs, pooling=None
    )
    base.trainable = False  # Start frozen

    x = base.output
    x = se_block(x, ratio=16)
    x = cbam_block(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Enhanced head
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='NutriScan_EfficientNetB3_v3')
    return model

# ── Data Pipeline ─────────────────────────────────────────────────
def make_dataset(subset, augment=False):
    path = os.path.join(DATA_DIR, subset)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        for cls in CLASS_NAMES: os.makedirs(os.path.join(path, cls), exist_ok=True)

    ds = keras.utils.image_dataset_from_directory(
        path, labels='inferred', label_mode='categorical',
        class_names=CLASS_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=(subset == 'train'), seed=42
    )

    preprocess = keras.applications.efficientnet.preprocess_input

    if augment:
        augmentor = keras.Sequential([
            keras.layers.RandomFlip('horizontal_and_vertical'),
            keras.layers.RandomRotation(0.25),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomBrightness(0.2),
            keras.layers.RandomContrast(0.2),
        ])
        ds = ds.map(lambda x, y: (augmentor(preprocess(x), training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (preprocess(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)

# ── Class weights ────────────────────────────────────────────────
def compute_class_weights(train_ds):
    counts = np.zeros(NUM_CLASSES)
    # Use as_numpy_iterator to check safely
    it = train_ds.as_numpy_iterator()
    try:
        first = next(it)
    except StopIteration:
        return {i: 1.0 for i in range(NUM_CLASSES)}
    
    for _, labels in train_ds: counts += labels.numpy().sum(axis=0)
    total = sum(counts)
    if total == 0: return {i: 1.0 for i in range(NUM_CLASSES)}
    return {i: total / (NUM_CLASSES * (counts[i] + 1e-6)) for i in range(NUM_CLASSES)}

# ── Confusion Matrix Plot ─────────────────────────────────────────
def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('NutriScan Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

# ── Main Training ─────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  NutriScan AI — TARGET 95% ACCURACY MODE (FINAL)")
    print("=" * 60)

    train_ds = make_dataset('train', augment=True)
    val_ds   = make_dataset('val',   augment=False)
    test_ds  = make_dataset('test',  augment=False)

    class_weights = compute_class_weights(train_ds)

    model = build_model()
    
    # ── PHASE 1: Warmup ──────────────
    print("\n[Phase 1] Head Initialization...")
    model.compile(optimizer=keras.optimizers.Adam(LR_P1),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    cb_p1 = [
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.weights.h5'),
                                     monitor='val_accuracy', save_best_only=True,
                                     save_weights_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history_p1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1, 
                          callbacks=cb_p1, class_weight=class_weights)

    # ── PHASE 2: Deep Fine-Tuning ──────────────
    print("\n[Phase 2] Deep Fine-Tuning Backbone (Target: 95%+)...")
    model.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(LR_P2),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    cb_p2 = [
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.weights.h5'),
                                     monitor='val_accuracy', save_best_only=True,
                                     save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    ]

    history_p2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P2, 
                          callbacks=cb_p2, class_weight=class_weights)

    # ── Final Evaluation & Confusion Matrix ───────
    print("\n[Final Evaluation]")
    if os.path.exists(os.path.join(MODEL_DIR, 'best_model.weights.h5')):
        model.load_weights(os.path.join(MODEL_DIR, 'best_model.weights.h5'))
    
    # 1. Plot Curves
    try:
        acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
        val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
        plt.figure(figsize=(10, 5))
        plt.plot(acc, label='Train Acc')
        plt.plot(val_acc, label='Val Acc')
        plt.axhline(0.95, color='green', linestyle='--', label='95% Target')
        plt.title('NutriScan Training History')
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")

    # 2. Confusion Matrix
    print("\nGenerating Detailed Report & Matrix...")
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

if __name__ == '__main__':
    train()
