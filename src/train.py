import os
import json
import cv2
import numpy as np

# ================================
# Paths (Anchored to Project Root)
# ================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "Dataset_Split", "train")
VAL_PATH = os.path.join(PROJECT_ROOT, "Dataset_Split", "val")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cricket_player_model.keras")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "cricket_player_model.npz")
HAAR_XML_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.json")

# ================================
# Settings
# ================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def detect_faces_safe(img_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)):
    """
    Detects faces using cv2.CascadeClassifier when available (OpenCV 4.x),
    and falls back to a multi-scale facial skin/gradient bounding box detector
    when running under OpenCV 5.0+ where CascadeClassifier is removed.
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "CascadeClassifier"):
        xml_path = HAAR_XML_PATH
        if not os.path.exists(xml_path) and hasattr(cv2, "data"):
            xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(xml_path):
            cascade = cv2.CascadeClassifier(xml_path)
            if not cascade.empty():
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scaleFactor,
                    minNeighbors=minNeighbors,
                    minSize=minSize
                )
                if len(faces) > 0:
                    return list(faces)

    # OpenCV 5.0+ fallback: skin-tone + facial edge density bounding box
    h, w = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= minSize[0] and ch >= minSize[1] and 0.45 <= (cw / float(ch + 1e-5)) <= 1.8:
            candidates.append((x, y, cw, ch))

    if candidates:
        return candidates

    # Portrait center box fallback so prediction never fails
    side = int(min(w, h) * 0.65)
    cx, cy = w // 2, int(h * 0.42)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return [(x1, y1, min(side, w - x1), min(side, h - y1))]


def _compute_numpy_hog(gray_128):
    """
    Pure NumPy + cv2.Sobel 9-bin spatial orientation histogram on a 128x128 grayscale face crop.
    Works identically on OpenCV 4.x and OpenCV 5.0+ without requiring cv2.HOGDescriptor.
    """
    gx = cv2.Sobel(gray_128, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_128, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)

    n_bins = 9
    bin_idx = np.clip((ang / (180.0 / n_bins)).astype(np.int32), 0, n_bins - 1)
    # 8x8 grid of 16x16 cells
    cells = np.zeros((8, 8, n_bins), dtype=np.float32)
    for b in range(n_bins):
        mask_mag = np.where(bin_idx == b, mag, 0.0)
        cells[:, :, b] = mask_mag.reshape(8, 16, 8, 16).sum(axis=(1, 3))

    # 2x2 cell block normalization with stride 1 -> 7x7 blocks of 36 features = 1764 features
    blocks = []
    for r in range(7):
        for c in range(7):
            blk = cells[r:r+2, c:c+2, :].flatten()
            blk = blk / (np.linalg.norm(blk) + 1e-6)
            blocks.append(blk)
    feat = np.concatenate(blocks)
    return feat / (np.linalg.norm(feat) + 1e-8)


def extract_face_features(img_bgr):
    """Multi-descriptor facial representation (Spatial Orientation Histogram + HSV/LAB + CLAHE template) compatible with OpenCV 4.x and 5.0+."""
    img = cv2.resize(img_bgr, IMAGE_SIZE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_128 = cv2.resize(gray_eq, (128, 128))
    hog_feat = _compute_numpy_hog(gray_128)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color_feats = []
    h_step, w_step = IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4
    for r in range(4):
        for c in range(4):
            patch_hsv = hsv[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            patch_lab = lab[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            hist_h = cv2.calcHist([patch_hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
            hist_l = cv2.calcHist([patch_lab], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256]).flatten()
            color_feats.append(hist_h / (np.linalg.norm(hist_h) + 1e-8))
            color_feats.append(hist_l / (np.linalg.norm(hist_l) + 1e-8))
    color_feat = np.concatenate(color_feats)
    color_feat = color_feat / (np.linalg.norm(color_feat) + 1e-8)

    small_eq = cv2.resize(gray_eq, (32, 32)).astype(np.float32).flatten()
    small_eq = (small_eq - np.mean(small_eq)) / (np.std(small_eq) + 1e-8)
    small_eq = small_eq / (np.linalg.norm(small_eq) + 1e-8)

    return np.concatenate([hog_feat * 1.2, color_feat * 1.0, small_eq * 0.6]).astype(np.float32)


def augment_image(img_bgr):
    """Replicates RandomFlip('horizontal'), RandomRotation(0.08), RandomZoom(0.10), RandomContrast(0.10)."""
    aug_list = [img_bgr, cv2.flip(img_bgr, 1)]
    h, w = img_bgr.shape[:2]
    for angle in (-6, 6):
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.05)
        aug_list.append(cv2.warpAffine(img_bgr, mat, (w, h), borderMode=cv2.BORDER_REFLECT))
    aug_list.append(cv2.convertScaleAbs(img_bgr, alpha=1.08, beta=5))
    return aug_list


def train_with_tensorflow():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    print("\nClass names:")
    for i, name in enumerate(class_names):
        print(i, name)

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ])

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("\n================================")
    print("STAGE 1: TRAINING CLASSIFIER")
    print("================================\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=STAGE1_EPOCHS,
        callbacks=[early_stop, checkpoint]
    )

    print("\n================================")
    print("STAGE 2: FINE TUNING MODEL")
    print("================================\n")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=STAGE2_EPOCHS,
        callbacks=[early_stop, checkpoint]
    )

    loss, accuracy = model.evaluate(validation_dataset)

    print("\n================================")
    print("TRAINING COMPLETED")
    print("================================")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Best model saved at: {MODEL_PATH}")
    print(f"Class names saved at: {CLASS_NAMES_PATH}")


def train_fallback():
    class_names = sorted([
        d for d in os.listdir(TRAIN_PATH)
        if os.path.isdir(os.path.join(TRAIN_PATH, d))
    ])
    print("\nClass names:")
    for i, name in enumerate(class_names):
        print(i, name)

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

    print("\n================================")
    print("STAGE 1 & 2: FEATURE EXTRACTION & CLASSIFIER TRAINING")
    print("================================\n")

    X_train, y_train = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(TRAIN_PATH, cname)
        for fname in sorted(os.listdir(cdir)):
            fpath = os.path.join(cdir, fname)
            img = cv2.imread(fpath)
            if img is not None:
                for aug_img in augment_image(img):
                    X_train.append(extract_face_features(aug_img))
                    y_train.append(idx)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    X_val, y_val = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(VAL_PATH, cname)
        for fname in sorted(os.listdir(cdir)):
            fpath = os.path.join(cdir, fname)
            img = cv2.imread(fpath)
            if img is not None:
                X_val.append(extract_face_features(img))
                y_val.append(idx)

    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)

    K_train = X_train @ X_train.T
    n_samples = X_train.shape[0]
    n_classes = len(class_names)

    counts = np.bincount(y_train)
    class_weights = (len(y_train) / (n_classes * counts)) ** 0.5
    Y_onehot = np.eye(n_classes, dtype=np.float32)[y_train] * class_weights[None, :]

    reg = 0.25
    alpha = np.linalg.solve(K_train + reg * np.eye(n_samples, dtype=np.float32), Y_onehot)
    W = (X_train.T @ alpha).astype(np.float32)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    b = np.zeros(n_classes, dtype=np.float32)

    val_logits = X_val @ W + b
    val_preds = np.argmax(val_logits, axis=1)
    accuracy = float(np.mean(val_preds == y_val))

    np.savez_compressed(
        FALLBACK_MODEL_PATH,
        W=W,
        b=b,
        temperature=np.array([25.0], dtype=np.float32),
        class_names=np.array(class_names)
    )

    print("\n================================")
    print("TRAINING COMPLETED")
    print("================================")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved at: {FALLBACK_MODEL_PATH}")
    print(f"Class names saved at: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    if TF_AVAILABLE:
        train_with_tensorflow()
    else:
        train_fallback()
