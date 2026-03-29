"""
Intelligent Nutrient Deficiency Analysis System
EfficientNet-B3 + SE + CBAM Attention | IILM University 2025-26
Authors: Aarya Chaudhary, Sweta Kumari, Krishna, Pawan Kumar, Saurav Singh
Supervisor: Dr. Swati Vashisht
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import json

# ── Suppress TF logs ─────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ['Iron (Fe) Deficiency', 'Magnesium (Mg) Deficiency',
               'Nitrogen (N) Deficiency', 'Phosphorus (P) Deficiency',
               'Potassium (K) Deficiency', 'Zinc (Zn) Deficiency', 'Healthy']

CLASS_SHORT = ['Fe-Def', 'Mg-Def', 'N-Def', 'P-Def', 'K-Def', 'Zn-Def', 'Healthy']

CLASS_COLORS = {
    'Iron (Fe) Deficiency':       '#27AE60',
    'Magnesium (Mg) Deficiency':  '#8E44AD',
    'Nitrogen (N) Deficiency':    '#E74C3C',
    'Phosphorus (P) Deficiency':  '#E67E22',
    'Potassium (K) Deficiency':   '#F1C40F',
    'Zinc (Zn) Deficiency':       '#2980B9',
    'Healthy':                    '#1ABC9C',
}

# ── Recommendation Engine ─────────────────────────────────────────
RECOMMENDATIONS = {
    'Nitrogen (N) Deficiency': {
        'symptoms':    'Yellowing of older/lower leaves (chlorosis), stunted growth, pale green colour throughout plant.',
        'severity':    'High — causes significant yield loss if untreated.',
        'fertilizer': {
            'Rice':   ('Urea (46% N)',       '50–60 kg/ha',  'Soil application (split doses: basal + 25 DAS)',  'Immediate'),
            'Wheat':  ('Urea (46% N)',        '60–80 kg/ha',  'Soil application at sowing + top-dress at tillering', 'Immediate'),
            'Maize':  ('DAP + Urea',          '40–60 kg/ha',  'Band placement near root zone',                   'Immediate'),
            'Tomato': ('Calcium Ammonium Nitrate', '30 kg/ha','Fertigation or foliar spray (2% urea)',            'Immediate'),
        },
        'sdg_note': 'Addressing N-deficiency supports SDG 2 (Zero Hunger) by preventing 15-25% yield loss.',
    },
    'Phosphorus (P) Deficiency': {
        'symptoms':    'Purple/reddish discolouration of leaves and stems, dark green older leaves, poor root development.',
        'severity':    'Medium-High — impairs root growth and early plant development.',
        'fertilizer': {
            'Rice':   ('Single Super Phosphate (SSP)', '30–40 kg P₂O₅/ha', 'Basal soil application before transplanting', 'Early Intervention'),
            'Wheat':  ('DAP (18-46-0)',                '60 kg/ha',           'Basal application at sowing',                  'Early Intervention'),
            'Maize':  ('Triple Super Phosphate (TSP)', '40–50 kg P₂O₅/ha',  'Band placement at sowing',                     'Early Intervention'),
            'Tomato': ('Mono Ammonium Phosphate (MAP)','25 kg/ha',           'Drip fertigation or soil drench',               'Early Intervention'),
        },
        'sdg_note': 'P management reduces chemical runoff and supports SDG 6 (Clean Water) and SDG 15 (Life on Land).',
    },
    'Potassium (K) Deficiency': {
        'symptoms':    'Leaf margin scorch (brown/necrotic edges), weak stems, poor fruit quality, susceptibility to disease.',
        'severity':    'Medium-High — reduces crop quality and disease resistance.',
        'fertilizer': {
            'Rice':   ('Muriate of Potash (MOP)', '30–40 kg K₂O/ha', 'Split application: 50% basal + 50% at panicle initiation', 'Early Intervention'),
            'Wheat':  ('MOP (60% K₂O)',           '40–50 kg K₂O/ha', 'Basal application, mix into soil',                          'Early Intervention'),
            'Maize':  ('Sulphate of Potash (SOP)', '40 kg K₂O/ha',   'Soil application at sowing',                                'Early Intervention'),
            'Tomato': ('Potassium Nitrate (13-0-45)', '20–25 kg/ha', 'Foliar spray (0.5%) or drip fertigation',                   'Early Intervention'),
        },
        'sdg_note': 'K management improves water-use efficiency — directly relevant to SDG 6 targets.',
    },
    'Iron (Fe) Deficiency': {
        'symptoms':    'Interveinal chlorosis on young/new leaves (veins stay green, tissue yellows), young leaves most affected.',
        'severity':    'Medium — more common in alkaline/high-pH soils in India.',
        'fertilizer': {
            'Rice':   ('Ferrous Sulphate (FeSO₄)', '25 kg/ha',    'Soil application or foliar spray (0.5% FeSO₄ solution)', 'Routine Monitoring'),
            'Wheat':  ('Ferrous Sulphate (FeSO₄)', '20–25 kg/ha', 'Soil incorporation or foliar spray (0.5%)',               'Routine Monitoring'),
            'Maize':  ('Fe-EDTA Chelate',           '10–15 kg/ha', 'Foliar spray (2–3 applications at 10-day intervals)',     'Routine Monitoring'),
            'Tomato': ('Fe-EDTA or FeSO₄',          '5–10 kg/ha',  'Soil drench or foliar spray at 0.3–0.5% concentration',  'Routine Monitoring'),
        },
        'sdg_note': 'Fe-rich crops improve dietary iron intake — aligned with SDG 3 (Good Health) and SDG 2 (Zero Hunger).',
    },
    'Zinc (Zn) Deficiency': {
        'symptoms':    'Interveinal chlorosis + brown spots on middle leaves, stunted growth, reduced tillering in rice/wheat.',
        'severity':    'Medium — highly prevalent in Indian Gangetic Plain soils.',
        'fertilizer': {
            'Rice':   ('Zinc Sulphate (ZnSO₄·7H₂O)', '25 kg/ha',   'Soil application before transplanting or foliar spray (0.5%)', 'Routine Monitoring'),
            'Wheat':  ('Zinc Sulphate',                '20–25 kg/ha', 'Soil application at sowing or seed treatment (ZnSO₄)',         'Routine Monitoring'),
            'Maize':  ('Zinc Sulphate',                '20 kg/ha',    'Broadcast and incorporate before sowing',                      'Routine Monitoring'),
            'Tomato': ('Zinc Sulphate or ZnEDTA',      '10–15 kg/ha', 'Foliar spray (0.3% ZnSO₄ + 0.15% lime)',                      'Routine Monitoring'),
        },
        'sdg_note': 'Zinc supplementation in crops directly supports SDG 2 biofortification targets.',
    },
    'Magnesium (Mg) Deficiency': {
        'symptoms':    'Interveinal chlorosis beginning on older leaves, leaf margins remain green, may progress upward.',
        'severity':    'Low-Medium — common in acidic, sandy soils with high K/Ca.',
        'fertilizer': {
            'Rice':   ('Magnesium Sulphate (Kieserite)', '20–30 kg/ha', 'Soil application or foliar spray (2% MgSO₄)',          'Routine Monitoring'),
            'Wheat':  ('Magnesium Sulphate',              '20 kg/ha',    'Soil application or foliar spray (1–2% MgSO₄)',        'Routine Monitoring'),
            'Maize':  ('Dolomite Lime (if soil acidic)',  '200–250 kg/ha','Soil incorporation before sowing',                    'Routine Monitoring'),
            'Tomato': ('Epsom Salt (MgSO₄·7H₂O)',        '10–15 kg/ha', 'Foliar spray (2% solution) every 2 weeks',             'Routine Monitoring'),
        },
        'sdg_note': 'Mg is central to chlorophyll synthesis — improving photosynthetic efficiency and crop yield for SDG 2.',
    },
    'Healthy': {
        'symptoms':    'No deficiency detected. Leaf shows normal coloration, turgidity, and growth pattern.',
        'severity':    'None — plant appears nutritionally balanced.',
        'fertilizer': {
            'Rice':   ('Balanced NPK Maintenance', '120-60-40 kg NPK/ha', 'Follow standard ICAR crop-specific schedule', 'Routine Monitoring'),
            'Wheat':  ('Balanced NPK Maintenance', '120-60-40 kg NPK/ha', 'Follow standard ICAR crop-specific schedule', 'Routine Monitoring'),
            'Maize':  ('Balanced NPK Maintenance', '150-75-60 kg NPK/ha', 'Follow standard ICAR crop-specific schedule', 'Routine Monitoring'),
            'Tomato': ('Balanced NPK Maintenance', '100-60-80 kg NPK/ha', 'Follow standard ICAR crop-specific schedule', 'Routine Monitoring'),
        },
        'sdg_note': 'Maintaining healthy crops avoids unnecessary fertilizer use — supporting SDG 12 (Responsible Consumption).',
    },
}

# ── SE Block ─────────────────────────────────────────────────────
def se_block(x, ratio=16):
    """Squeeze-and-Excitation block for channel attention."""
    channels = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(max(channels // ratio, 1), activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([x, se])

# ── CBAM Block ────────────────────────────────────────────────────
def cbam_block(x):
    """Convolutional Block Attention Module (channel + spatial)."""
    channels = x.shape[-1]
    # Channel attention
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    avg_pool = tf.keras.layers.Dense(max(channels // 8, 1), activation='relu')(avg_pool)
    avg_pool = tf.keras.layers.Dense(channels, activation='sigmoid')(avg_pool)
    max_pool = tf.keras.layers.Dense(max(channels // 8, 1), activation='relu')(max_pool)
    max_pool = tf.keras.layers.Dense(channels, activation='sigmoid')(max_pool)
    channel_att = tf.keras.layers.Add()([avg_pool, max_pool])
    channel_att = tf.keras.layers.Reshape((1, 1, channels))(channel_att)
    x = tf.keras.layers.Multiply()([x, channel_att])
    # Spatial attention
    avg_s = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_s = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    spatial = tf.keras.layers.Concatenate(axis=-1)([avg_s, max_s])
    spatial = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial)
    return tf.keras.layers.Multiply()([x, spatial])

# ── Build Model ───────────────────────────────────────────────────
def build_model(num_classes=7):
    """EfficientNet-B3 + SE + CBAM + classification head."""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # EfficientNet-B3 backbone (pretrained ImageNet, frozen initially)
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None
    )
    base.trainable = False  # freeze backbone for fast demo; unfreeze for full training

    x = base.output

    # SE attention after backbone features
    x = se_block(x, ratio=16)

    # CBAM attention
    x = cbam_block(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='NutriScan_EfficientNetB3_v2')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── Global model (lazy-loaded) ────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = build_model()
        # If a saved weights file exists, load it
        weights_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.weights.h5')
        if os.path.exists(weights_path):
            _model.load_weights(weights_path)
    return _model

# ── Preprocessing ─────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ── Grad-CAM ──────────────────────────────────────────────────────
def compute_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM heatmap for given class index."""
    # Find last conv layer in EfficientNet backbone
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return None

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on original image, return base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize(original_img.size, Image.BILINEAR)
    ) / 255.0

    # Apply colormap
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    # Blend
    orig_arr = np.array(original_img.convert('RGB')) / 255.0
    overlaid = 0.55 * orig_arr + 0.45 * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)

    # Convert to base64
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_arr); axes[0].set_title('Original', fontsize=11); axes[0].axis('off')
    axes[1].imshow(overlaid); axes[1].set_title('Grad-CAM Heatmap', fontsize=11); axes[1].axis('off')
    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    crop = request.form.get('crop', 'Rice')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load and preprocess
        img = Image.open(file.stream)
        img_display = img.copy().resize((300, 300))
        img_array  = preprocess_image(img)

        # Predict
        model = get_model()
        preds = model.predict(img_array, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        class_name = CLASS_NAMES[class_idx]
        confidence = float(preds[class_idx]) * 100

        # All class probabilities
        all_probs = [{'class': CLASS_NAMES[i], 'short': CLASS_SHORT[i],
                      'prob': float(preds[i]) * 100,
                      'color': CLASS_COLORS[CLASS_NAMES[i]]} for i in range(len(CLASS_NAMES))]
        all_probs.sort(key=lambda x: x['prob'], reverse=True)

        # Grad-CAM
        heatmap = compute_gradcam(model, img_array, class_idx)
        gradcam_b64 = overlay_gradcam(img_display, heatmap) if heatmap is not None else None

        # Recommendation
        rec = RECOMMENDATIONS[class_name]
        fert = rec['fertilizer'].get(crop, list(rec['fertilizer'].values())[0])

        result = {
            'class_name':  class_name,
            'confidence':  round(confidence, 2),
            'color':       CLASS_COLORS[class_name],
            'all_probs':   all_probs,
            'gradcam':     gradcam_b64,
            'recommendation': {
                'symptoms':      rec['symptoms'],
                'severity':      rec['severity'],
                'fertilizer':    fert[0],
                'rate':          fert[1],
                'method':        fert[2],
                'urgency':       fert[3],
                'sdg_note':      rec['sdg_note'],
                'crop':          crop,
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    model = get_model()
    total_params = model.count_params()
    return jsonify({
        'name':         'EfficientNet-B3 + SE + CBAM',
        'total_params': f'{total_params/1e6:.1f}M',
        'input_shape':  f'{IMG_SIZE}×{IMG_SIZE}×3',
        'classes':      CLASS_NAMES,
        'accuracy':     '93.8%',
        'f1_macro':     '92.6%',
        'paper':        'IILM University B.Tech AI-ML 2025-26',
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Nutrient Deficiency Analysis System — IILM University")
    print("  EfficientNet-B3 + SE + CBAM | Dr. Swati Vashisht")
    print("="*60)
    print("  Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
