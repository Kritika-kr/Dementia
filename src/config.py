"""Central configuration: paths, constants, random seed."""
from pathlib import Path

RANDOM_SEED = 42

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "transcripts.csv"          # columns: speaker_id, text, label
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
VECTORIZER_DIR = OUTPUT_DIR / "vectorizers"

for d in (MODEL_DIR, VECTORIZER_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Feature extraction ---
TFIDF_MAX_FEATURES = 300
COUNT_MAX_FEATURES = 300
WORD2VEC_DIM = 100
GLOVE_MODEL_NAME = "glove-wiki-gigaword-300"
SPACY_MODEL_NAME = "en_core_web_md"
HESITATION_WORDS = {"uh", "um", "erm", "ah", "eh", "hmm"}

# --- Splitting ---
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Labels ---
LABEL_MAP = {0: "Control", 1: "Dementia"}
