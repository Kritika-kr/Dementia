"""
Reports Accuracy / Precision / Recall / F1 / ROC-AUC on the held-out
test set, plus a feature-block ablation study for the paper's results
table (justifies combining all 7 blocks instead of asserting it).
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from . import config
from .train_pipeline import build_pipeline


def score(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def ablation_study(train_blocks, y_train, test_blocks, y_test) -> pd.DataFrame:
    """Train the same ensemble type on different feature-block subsets."""
    from .feature_extraction import FeatureExtractor

    subsets = {
        "lexical_only": ["tfidf", "count"],
        "embedding_only": ["word2vec", "glove", "spacy_sent_embed"],
        "linguistic_only": ["hesitation", "linguistic"],
        "graph_only": ["graph"],
        "lexical_plus_embedding": ["tfidf", "count", "word2vec", "glove", "spacy_sent_embed"],
        "full_combination": list(train_blocks.keys()),
    }

    rows = []
    for label, keys in subsets.items():
        X_tr = FeatureExtractor.concat_blocks(train_blocks, use=keys)
        X_te = FeatureExtractor.concat_blocks(test_blocks, use=keys)
        k = min(300, X_tr.shape[1])  # can't select more features than exist in this subset

        ensemble = VotingClassifier(
            estimators=[
                ("l1_logreg", build_pipeline(
                    LogisticRegression(penalty="l1", solver="liblinear",
                                        max_iter=2000, random_state=config.RANDOM_SEED), k_features=k)),
                ("random_forest", build_pipeline(
                    RandomForestClassifier(random_state=config.RANDOM_SEED), k_features=k)),
                ("svm_rbf", build_pipeline(
                    SVC(kernel="rbf", probability=True, random_state=config.RANDOM_SEED), k_features=k)),
            ],
            voting="soft",
        )
        ensemble.fit(X_tr, y_train)
        metrics = score(ensemble, X_te, y_test)
        metrics["feature_set"] = label
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("feature_set")


if __name__ == "__main__":
    from .train_pipeline import main as train_main
    ensemble, X_test, y_test, _ = train_main()
    print("Held-out test metrics:")
    for k, v in score(ensemble, X_test, y_test).items():
        print(f"  {k}: {v:.3f}")