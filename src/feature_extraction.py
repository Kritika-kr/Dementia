"""
Batch feature extraction. All vectorizers/embedding lookups are FIT
on the training set only; the test set is only ever transformed.

Features are kept as a dict of named blocks (not concatenated) so an
ablation study can later use any subset of blocks.
"""
import re
import joblib
import numpy as np
import pandas as pd
import spacy
import gensim.downloader as gensim_api
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from . import config
from .graph_features import graph_features_batch

_nlp = None
try:
    _nlp = spacy.load(config.SPACY_MODEL_NAME)
except OSError:
    # Model wheel wasn't picked up from requirements.txt (e.g. a fresh
    # deployment environment) — download it once at startup as a fallback.
    from spacy.cli import download as _spacy_download
    _spacy_download(config.SPACY_MODEL_NAME)
    _nlp = spacy.load(config.SPACY_MODEL_NAME)


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))
    return text.lower()


def count_hesitations(text: str) -> int:
    return sum(1 for w in text.lower().split() if w in config.HESITATION_WORDS)


def linguistic_features(text: str) -> list:
    doc = _nlp(text)
    n_tokens = len(doc)
    n_sents = len(list(doc.sents))
    avg_len = np.mean([len(t.text) for t in doc]) if n_tokens else 0.0
    n_noun = sum(1 for t in doc if t.pos_ == "NOUN")
    n_verb = sum(1 for t in doc if t.pos_ == "VERB")
    return [n_tokens, n_sents, avg_len, n_noun, n_verb]


def avg_embedding(text: str, kv_model, dim: int) -> np.ndarray:
    words = [w for w in text.split() if w in kv_model]
    if not words:
        return np.zeros(dim)
    return np.mean([kv_model[w] for w in words], axis=0)


class FeatureExtractor:
    """Fit on train texts once; reuse to transform any split."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES)
        self.count = CountVectorizer(max_features=config.COUNT_MAX_FEATURES)
        self.word2vec = None   # set via fit_word2vec or load
        self.glove = None

    def fit(self, train_texts: pd.Series, word2vec_model):
        cleaned = train_texts.apply(clean_text)
        self.tfidf.fit(cleaned)
        self.count.fit(cleaned)
        self.word2vec = word2vec_model
        self.glove = gensim_api.load(config.GLOVE_MODEL_NAME)
        return self

    def transform(self, texts: pd.Series) -> dict:
        """Returns a dict of named feature-block arrays, shape (n_samples, dim)."""
        cleaned = texts.apply(clean_text)

        blocks = {
            "tfidf": self.tfidf.transform(cleaned).toarray(),
            "count": self.count.transform(cleaned).toarray(),
            "word2vec": np.vstack(
                [avg_embedding(t, self.word2vec, config.WORD2VEC_DIM) for t in cleaned]
            ),
            "glove": np.vstack(
                [avg_embedding(t, self.glove, 300) for t in cleaned]
            ),
            "spacy_sent_embed": np.vstack([_nlp(t).vector for t in cleaned]),
            "hesitation": np.array([[count_hesitations(t)] for t in cleaned]),
            "linguistic": np.vstack([linguistic_features(t) for t in cleaned]),
            "graph": graph_features_batch(cleaned),
        }
        return blocks

    @staticmethod
    def concat_blocks(blocks: dict, use: list = None) -> np.ndarray:
        """Concatenate selected feature blocks (default: all) into one matrix."""
        keys = use or list(blocks.keys())
        return np.concatenate([blocks[k] for k in keys], axis=1)

    def save(self):
        joblib.dump(self.tfidf, config.VECTORIZER_DIR / "tfidf_vectorizer.pkl")
        joblib.dump(self.count, config.VECTORIZER_DIR / "count_vectorizer.pkl")
        joblib.dump(self.glove, config.VECTORIZER_DIR / "glove_keyedvectors.pkl")
        self.word2vec.save(str(config.MODEL_DIR / "word2vec_model.bin"))