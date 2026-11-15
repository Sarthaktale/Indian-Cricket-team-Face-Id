# Indian Cricket Team Member Face Identification

An AI-based computer vision project replicated from [harshinimadasu17-design/Indian-Cricket-Team-Face-Identification](https://github.com/harshinimadasu17-design/Indian-Cricket-Team-Face-Identification/tree/main) that identifies Indian cricket players from facial images using OpenCV Haar Cascade face detection, MobileNetV2 transfer learning (with a built-in OpenCV/NumPy multi-descriptor facial embedding fallback for zero-dependency execution), and a Streamlit web interface.

## Problem Statement
Develop a machine learning / computer vision system that identifies an Indian cricket team member from an image based on facial features. The system detects and crops faces using OpenCV's Haar Cascade classifier, preprocesses images to `224 × 224`, trains a 15-class classifier on the provided dataset (`archive (2).zip`), and classifies new input images with top-3 confidence scores.

## Players / Classes (15 Players from Provided Dataset)
The model is trained to classify **15 Indian cricket players** from the uploaded dataset:

1. Bhuvneshwar Kumar (`bhuvneshwar_kumar`)
2. Dinesh Karthik (`dinesh_karthik`)
3. Hardik Pandya (`hardik_pandya`)
4. Jasprit Bumrah (`jasprit_bumrah`)
5. KL Rahul (`k._l._rahul`)
6. Kedar Jadhav (`kedar_jadhav`)
7. Kuldeep Yadav (`kuldeep_yadav`)
8. Mohammed Shami (`mohammed_shami`)
9. MS Dhoni (`ms_dhoni`)
10. Ravindra Jadeja (`ravindra_jadeja`)
11. Rohit Sharma (`rohit_sharma`)
12. Shikhar Dhawan (`shikhar_dhawan`)
13. Vijay Shankar (`vijay_shankar`)
14. Virat Kohli (`virat_kohli`)
15. Yuzvendra Chahal (`yuzvendra_chahal`)

## Technologies Used
- **Python 3**
- **OpenCV (`cv2`)** — Haar Cascade frontal face detection (`haarcascade_frontalface_default.xml`), CLAHE, HOG & spatial color descriptors
- **TensorFlow / Keras (`MobileNetV2`)** — Two-stage transfer learning & fine-tuning (`models/cricket_player_model.keras`), with automatic fallback to a pre-trained L2-regularized cosine-softmax facial descriptor model (`models/cricket_player_model.npz`) when TensorFlow is not installed
- **NumPy & SciPy** — Feature normalization and linear/softmax classification
- **Pillow (`PIL`)** — Image handling
- **Scikit-learn & Matplotlib** — Classification report and confusion matrix visualization
- **Streamlit** — Interactive web application (`app.py`)

## System Workflow
`Input Image -> Haar Cascade Face Detection -> Largest Face Cropping (+20px margin) -> 224x224 Preprocessing & Augmentation -> Feature Extraction (MobileNetV2 / Multi-Descriptor Embedding) -> 15-Class Softmax Classification -> Player Name + Top-3 Confidence`

## Dataset Summary (Prepared from Uploaded `archive (2).zip`)
- **Total raw images (`Dataset/`)**: `576` images across `15` player directories (plus `players.csv`)
- **Haar Cascade face detection (`Prepare_dataset.py`)**:
  - **Faces successfully cropped & saved (`Dataset_Faces/`)**: `412` face images
  - **No frontal face detected**: `164` images
  - **Invalid/corrupt images**: `0`
- **Balanced 80/20 Train/Validation Split (`create_split.py`, `seed=42`)**:
  - **Training set (`Dataset_Split/train/`)**: `324` face images
  - **Validation set (`Dataset_Split/val/`)**: `88` face images

| Player Class | Raw Images (`Dataset/`) | Extracted Faces (`Dataset_Faces/`) | Train Split (`80%`) | Validation Split (`20%`) | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bhuvneshwar_kumar` | 37 | 27 | 21 | 6 | 50.00% (3/6) |
| `dinesh_karthik` | 32 | 20 | 16 | 4 | 75.00% (3/4) |
| `hardik_pandya` | 40 | 23 | 18 | 5 | 100.00% (5/5) |
| `jasprit_bumrah` | 21 | 12 | 9 | 3 | 33.33% (1/3) |
| `k._l._rahul` | 36 | 28 | 22 | 6 | 66.67% (4/6) |
| `kedar_jadhav` | 37 | 29 | 23 | 6 | 66.67% (4/6) |
| `kuldeep_yadav` | 45 | 27 | 21 | 6 | 33.33% (2/6) |
| `mohammed_shami` | 35 | 19 | 15 | 4 | 25.00% (1/4) |
| `ms_dhoni` | 46 | 37 | 29 | 8 | 62.50% (5/8) |
| `ravindra_jadeja` | 49 | 38 | 30 | 8 | 62.50% (5/8) |
| `rohit_sharma` | 50 | 40 | 32 | 8 | 75.00% (6/8) |
| `shikhar_dhawan` | 40 | 29 | 23 | 6 | 66.67% (4/6) |
| `vijay_shankar` | 31 | 21 | 16 | 5 | 40.00% (2/5) |
| `virat_kohli` | 48 | 43 | 34 | 9 | 66.67% (6/9) |
| `yuzvendra_chahal` | 29 | 19 | 15 | 4 | 75.00% (3/4) |
| **Total / Overall** | **576** | **412** | **324** | **88** | **61.36% (54/88)** |

## Project Structure
```text
Indian-Cricket-Team-Face-Identification/
├── .vscode/
│   └── settings.json
├── Dataset/
├── Dataset_Faces/
├── Dataset_Split/
│   ├── train/
│   └── val/
├── models/
│   └── cricket_player_model.npz  (and cricket_player_model.keras when trained with TensorFlow)
├── src/
│   ├── face detection.py
│   ├── predict.py
│   └── train.py
├── test_images/
│   ├── virat_2.jpg
│   ├── detected_face.jpg
│   └── virat_test.png
├── Prepare_dataset.py
├── README.md
├── app.py
├── check_dataset.py
├── class_names.json
├── confusion_matrix.png
├── create_split.py
├── evaluate.py
├── evaluation_results.txt
├── players.csv
└── requirements.txt
```

## Installation & Execution (Windows / Git Bash / Linux)
```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # On Windows Git Bash (or: venv\Scripts\activate on CMD)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline scripts in order (already executed and populated in this archive)
python Prepare_dataset.py
python check_dataset.py
python create_split.py
python src/train.py
python evaluate.py
python src/predict.py

# 4. Launch the Streamlit Web Application
streamlit run app.py
```
