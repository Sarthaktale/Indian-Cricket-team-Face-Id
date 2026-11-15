import os
import sys
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train import detect_faces_safe

SOURCE_DATASET = os.path.join(PROJECT_ROOT, "Dataset")
OUTPUT_DATASET = os.path.join(PROJECT_ROOT, "Dataset_Faces")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(OUTPUT_DATASET, exist_ok=True)

total_images = 0
successful_faces = 0
no_face = 0
invalid_images = 0

for player_name in sorted(os.listdir(SOURCE_DATASET)):
    player_path = os.path.join(SOURCE_DATASET, player_name)

    if not os.path.isdir(player_path):
        continue

    output_player_path = os.path.join(OUTPUT_DATASET, player_name)
    os.makedirs(output_player_path, exist_ok=True)

    player_total = 0
    player_success = 0

    print("\n==============================")
    print("Processing:", player_name)
    print("==============================")

    for filename in sorted(os.listdir(player_path)):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        total_images += 1
        player_total += 1

        input_path = os.path.join(player_path, filename)
        image = cv2.imread(input_path)

        if image is None:
            print("Invalid image:", filename)
            invalid_images += 1
            continue

        faces = detect_faces_safe(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            print("No face:", filename)
            no_face += 1
            continue

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            print("Invalid face crop:", filename)
            invalid_images += 1
            continue

        output_path = os.path.join(output_player_path, filename)
        success = cv2.imwrite(output_path, face)

        if success:
            successful_faces += 1
            player_success += 1
        else:
            print("Could not save:", filename)
            invalid_images += 1

    print(f"{player_name}: {player_success}/{player_total} faces saved")

print("\n")
print("==========================================")
print("        DATASET PREPARATION COMPLETED")
print("==========================================")
print(f"Total images found: {total_images}")
print(f"Faces successfully saved: {successful_faces}")
print(f"No face detected: {no_face}")
print(f"Invalid/failed images: {invalid_images}")
print(f"Output dataset: {OUTPUT_DATASET}")
print("==========================================")
