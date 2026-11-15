import os
import sys
import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from train import detect_faces_safe


def detect_face(image_path):
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None

    faces = detect_faces_safe(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print("No face detected.")
        return None

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    face = image[y:y+h, x:x+w]
    return face


if __name__ == "__main__":
    image_path = os.path.join(PROJECT_ROOT, "test_images", "virat_2.jpg")
    face = detect_face(image_path)

    if face is not None:
        out_path = os.path.join(PROJECT_ROOT, "test_images", "detected_face.jpg")
        cv2.imwrite(out_path, face)
        print("Face detected and saved.")
