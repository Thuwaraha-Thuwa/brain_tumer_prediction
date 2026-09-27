"""All paths and settings in one place."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"                   # downloaded .zip / .mat files
IMAGES_DIR = ROOT / "data" / "images"             # original MRI slices as PNG
MASKS_DIR = ROOT / "data" / "masks"               # tumour outlines drawn by doctors
ENHANCED_DIR = ROOT / "data" / "enhanced"         # after ODTWCHE contrast enhancement
SPLITS_DIR = ROOT / "data" / "splits"             # train.csv / val.csv / test.csv
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for _d in (RAW_DIR, IMAGES_DIR, MASKS_DIR, ENHANCED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset
# Figshare brain tumor dataset (Cheng et al. 2017) - used in the paper
FIGSHARE_API = "https://api.figshare.com/v2/articles/1512427"
TUMOR_TYPES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
# paper classifies benign vs malignant: glioma is malignant, the others benign
TUMOR_TO_CLASS = {1: 0, 2: 1, 3: 0}
CLASS_NAMES = ["benign", "malignant"]

IMAGES_PER_CLASS = 300        # small subset for CPU training (None = all 3064 images)
MAX_SLICES_PER_PATIENT = 5    # more patients = more variety for the model
SEED = 42

# ---------------------------------------------------------------- training
# Values from paper Table 3, shortened for CPU (paper: 30 epochs, LR drop every 10)
IMAGE_SIZE = 299              # Inception V3 input size
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_DROP_EVERY = 4             # multiply learning rate by 0.1 every N epochs
DROPOUT = 0.2
