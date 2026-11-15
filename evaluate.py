import os
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ============================================
# SETTINGS
# ============================================
TRAIN_PATH = "Dataset_Split/train"
VAL_PATH = "Dataset_Split/val"
MODEL_PATH = "models/cricket_player_model.keras"
FALLBACK_MODEL_PATH = "models/cricket_player_model.npz"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

try:
    import tensorflow as tf
    TF_AVAILABLE = os.path.exists(MODEL_PATH)
except ImportError:
    TF_AVAILABLE = False

# ============================================
# LOAD CLASS NAMES
# ============================================
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

# ============================================
# LOAD MODEL & GENERATE PREDICTIONS
# ============================================
print("\nLoading model and validation dataset...")
y_true = []
y_pred = []

if TF_AVAILABLE:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )
    print("\nGenerating predictions...")
    for images, labels in validation_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)
else:
    from train import extract_face_features
    weights = np.load(FALLBACK_MODEL_PATH, allow_pickle=True)
    W_model = weights["W"]
    b_model = weights["b"]
    print("Model loaded successfully.")
    print("\nGenerating predictions...")
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(VAL_PATH, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            fpath = os.path.join(cdir, fname)
            img = cv2.imread(fpath)
            if img is not None:
                feat = extract_face_features(img)
                logits = feat @ W_model + b_model
                y_true.append(idx)
                y_pred.append(int(np.argmax(logits)))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================
# ACCURACY
# ============================================
accuracy = np.mean(y_true == y_pred)

print("\n============================================")
print("MODEL EVALUATION")
print("============================================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# ============================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================
n_classes = len(class_names)
cm = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")
header = f"{'Player':<22} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
print(header)
print("-" * len(header))
precisions, recalls, f1s, supports = [], [], [], []
for i, cname in enumerate(class_names):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    supp = int(np.sum(cm[i, :]))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    supports.append(supp)
    print(f"{cname:<22} {prec:>10.2f} {rec:>10.2f} {f1:>10.2f} {supp:>10d}")

total_supp = int(np.sum(supports))
print("-" * len(header))
print(f"{'accuracy':<22} {'':>10} {'':>10} {accuracy:>10.2f} {total_supp:>10d}")
print(f"{'macro avg':<22} {np.mean(precisions):>10.2f} {np.mean(recalls):>10.2f} {np.mean(f1s):>10.2f} {total_supp:>10d}")
w_prec = np.average(precisions, weights=supports)
w_rec = np.average(recalls, weights=supports)
w_f1 = np.average(f1s, weights=supports)
print(f"{'weighted avg':<22} {w_prec:>10.2f} {w_rec:>10.2f} {w_f1:>10.2f} {total_supp:>10d}")

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")
print(cm)

# ============================================
# DISPLAY & SAVE CONFUSION MATRIX
# ============================================
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
display_labels = [c.replace("_", " ").title() for c in class_names]
ax.set(
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=display_labels,
    yticklabels=display_labels,
    title=f"Indian Cricket Player Face Identification - Confusion Matrix\n(Validation Accuracy: {accuracy * 100:.2f}%, n = {total_supp} faces across {n_classes} players)",
    ylabel="True Player",
    xlabel="Predicted Player"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm.max() / 2.0
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9, fontweight="bold" if i == j and cm[i, j] > 0 else "normal"
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close(fig)

# ============================================
# FIND CONFUSED PLAYER PAIRS
# ============================================
print("\n============================================")
print("MOST CONFUSED PLAYER PAIRS")
print("============================================")

confusions = []
for actual in range(n_classes):
    for predicted in range(n_classes):
        if actual != predicted:
            count = int(cm[actual][predicted])
            if count > 0:
                confusions.append((count, class_names[actual], class_names[predicted]))

confusions.sort(reverse=True)

if len(confusions) == 0:
    print("No incorrect predictions found.")
else:
    for count, actual, predicted in confusions:
        print(f"{actual} -> {predicted}: {count} images")

# ============================================
# PER-PLAYER ACCURACY
# ============================================
print("\n============================================")
print("PER-PLAYER ACCURACY")
print("============================================")

for i, player in enumerate(class_names):
    total_images = int(np.sum(cm[i]))
    correct_images = int(cm[i][i])
    if total_images > 0:
        player_accuracy = (correct_images / total_images) * 100
        print(f"{player}: {player_accuracy:.2f}% ({correct_images}/{total_images})")

# ============================================
# SAVE RESULTS
# ============================================
with open("evaluation_results.txt", "w") as file:
    file.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n\n")

    file.write("MOST CONFUSED PLAYER PAIRS\n")
    file.write("==========================\n")
    for count, actual, predicted in confusions:
        file.write(f"{actual} -> {predicted}: {count} images\n")

    file.write("\nPER-PLAYER ACCURACY\n")
    file.write("===================\n")
    for i, player in enumerate(class_names):
        total_images = int(np.sum(cm[i]))
        correct_images = int(cm[i][i])
        if total_images > 0:
            player_accuracy = (correct_images / total_images) * 100
            file.write(f"{player}: {player_accuracy:.2f}% ({correct_images}/{total_images})\n")

print("\n============================================")
print("EVALUATION COMPLETED")
print("============================================")
print("Confusion matrix saved as:\nconfusion_matrix.png")
print("\nDetailed results saved as:\nevaluation_results.txt")
