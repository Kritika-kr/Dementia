"""Loads saved artifacts and extracts features for ONE transcript at inference time.

IMPORTANT: this must extract exactly the same feature blocks, in the same
order, as generate_paper_assets.py's FINAL_FEATURE_SET used for training
(tfidf, count, word2vec, glove, spacy_sent_embed) — a mismatch here causes
either a shape error or silently wrong predictions.
"""
import joblib
import numpy as np
import spacy
from gensim.models import KeyedVectors
from . import config
from .feature_extraction import clean_text, avg_embedding

_nlp = spacy.load(config.SPACY_MODEL_NAME)
_tfidf = joblib.load(config.VECTORIZER_DIR / "tfidf_vectorizer.pkl")
_count = joblib.load(config.VECTORIZER_DIR / "count_vectorizer.pkl")
_glove = joblib.load(config.VECTORIZER_DIR / "glove_keyedvectors.pkl")
_word2vec = KeyedVectors.load(str(config.MODEL_DIR / "word2vec_model.bin"))
_ensemble = joblib.load(config.MODEL_DIR / "ensemble_l1logreg_rf_svm.pkl")


def extract_features(text: str) -> np.ndarray:
    cleaned = clean_text(text)
    blocks = [
        _tfidf.transform([cleaned]).toarray(),
        _count.transform([cleaned]).toarray(),
        avg_embedding(cleaned, _word2vec, config.WORD2VEC_DIM).reshape(1, -1),
        avg_embedding(cleaned, _glove, 300).reshape(1, -1),
        _nlp(cleaned).vector.reshape(1, -1),
    ]
    return np.concatenate(blocks, axis=1)


def predict(text: str) -> dict:
    features = extract_features(text)
    pred = _ensemble.predict(features)[0]
    proba = _ensemble.predict_proba(features)[0]
    return {
        "label": config.LABEL_MAP[pred],
        "confidence": {config.LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
    }