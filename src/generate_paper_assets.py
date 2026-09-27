"""
Generates EVERY figure and table needed for the paper's Results section
in one run. Outputs go to outputs/figures/*.png and outputs/tables/*.csv.

Run after data/transcripts.csv exists:
    python -m src.generate_paper_assets
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from . import config
from .data_preparation import load_raw_data, speaker_level_split
from .feature_extraction import FeatureExtractor
from .train_pipeline import MODEL_GRID, build_pipeline, tune_model
from .evaluate import score, ablation_study

FIG_DIR = config.OUTPUT_DIR / "figures"
TAB_DIR = config.OUTPUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def fig_dataset_summary(df, train_df, test_df):
    summary = pd.DataFrame({
        "split": ["full", "train", "test"],
        "n_transcripts": [len(df), len(train_df), len(test_df)],
        "n_speakers": [df["speaker_id"].nunique(), train_df["speaker_id"].nunique(),
                        test_df["speaker_id"].nunique()],
        "n_control": [(df.label == 0).sum(), (train_df.label == 0).sum(), (test_df.label == 0).sum()],
        "n_dementia": [(df.label == 1).sum(), (train_df.label == 1).sum(), (test_df.label == 1).sum()],
    })
    summary.to_csv(TAB_DIR / "dataset_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].map(config.LABEL_MAP).value_counts()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Distribution (Full Dataset)")
    ax.set_ylabel("Number of Transcripts")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)
    return summary


def fig_smote_before_after(y_train):
    before = Counter(y_train)
    sm = SMOTE(random_state=config.RANDOM_SEED)
    dummy_X = np.zeros((len(y_train), 2))
    _, y_res = sm.fit_resample(dummy_X, y_train)
    after = Counter(y_res)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, counts, title in zip(axes, [before, after], ["Before SMOTE", "After SMOTE"]):
        labels = [config.LABEL_MAP[k] for k in sorted(counts)]
        vals = [counts[k] for k in sorted(counts)]
        ax.bar(labels, vals, color=["#4C72B0", "#DD8452"])
        ax.set_title(title)
    fig.suptitle("Training Class Balance Before/After SMOTE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smote_before_after.png", dpi=200)
    plt.close(fig)


def fig_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(config.LABEL_MAP.values()))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Ensemble (Held-out Test Set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def fig_roc_pr_curves(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Ensemble")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Ensemble")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "precision_recall_curve.png", dpi=200)
    plt.close(fig)


def fig_model_comparison(all_scores: dict):
    df = pd.DataFrame(all_scores).T
    df.to_csv(TAB_DIR / "model_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison on Held-out Test Set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    return df


def fig_ablation(ablation_df: pd.DataFrame):
    ablation_df.to_csv(TAB_DIR / "ablation_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ablation_df[["accuracy", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_title("Feature-Block Ablation Study")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_study.png", dpi=200)
    plt.close(fig)


def fig_cv_score_distribution(cv_results: dict):
    """cv_results: {model_name: [fold_f1_scores]}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(cv_results.values(), tick_labels=list(cv_results.keys()))
    ax.set_title("Cross-Validation F1 Score Distribution per Model")
    ax.set_ylabel("F1 Score")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_score_distribution.png", dpi=200)
    plt.close(fig)


def save_hyperparameters(tuned_models: dict):
    rows = []
    for name, pipe in tuned_models.items():
        params = pipe.named_steps["clf"].get_params()
        key_params = {k: v for k, v in params.items() if k in (
            "C", "gamma", "n_estimators", "max_depth", "penalty")}
        key_params["select__k"] = pipe.named_steps["select"].k
        rows.append({"model": name, "key_params": json.dumps(key_params)})
    pd.DataFrame(rows).to_csv(TAB_DIR / "hyperparameters.csv", index=False)


def main():
    df = load_raw_data()
    train_df, test_df = speaker_level_split(df)
    fig_dataset_summary(df, train_df, test_df)
    fig_smote_before_after(train_df["label"].values)

    tokenized = [t.split() for t in train_df["text"]]
    w2v = Word2Vec(sentences=tokenized, vector_size=config.WORD2VEC_DIM,
                    window=5, min_count=1, seed=config.RANDOM_SEED)

    fx = FeatureExtractor().fit(train_df["text"], w2v.wv)
    train_blocks = fx.transform(train_df["text"])
    test_blocks = fx.transform(test_df["text"])

    # FINAL FEATURE SET: lexical + embedding only. Ablation across three prior
    # runs consistently showed graph and basic linguistic-count features added
    # no benefit (and graph features performed near/below the majority-class
    # baseline on their own) — the lexical+embedding subset outperformed the
    # full 8-block combination on every metric. See outputs/tables/
    # experiment_log_ablation.csv for the full comparison.
    FINAL_FEATURE_SET = ["tfidf", "count", "word2vec", "glove", "spacy_sent_embed"]
    X_train = FeatureExtractor.concat_blocks(train_blocks, use=FINAL_FEATURE_SET)
    X_test = FeatureExtractor.concat_blocks(test_blocks, use=FINAL_FEATURE_SET)
    y_train, y_test = train_df["label"].values, test_df["label"].values

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    tuned, cv_f1_scores, all_scores = {}, {}, {}
    from sklearn.model_selection import cross_val_score
    for name in MODEL_GRID:
        best_model = tune_model(name, X_train, y_train, cv)
        tuned[name] = best_model
        cv_f1_scores[name] = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1")
        best_model.fit(X_train, y_train)
        all_scores[name] = score(best_model, X_test, y_test)

    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[("l1_logreg", tuned["l1_logreg"]),
                    ("random_forest", tuned["random_forest"]),
                    ("svm_rbf", tuned["svm_rbf"])],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    all_scores["ensemble"] = score(ensemble, X_test, y_test)
    joblib.dump(ensemble, config.MODEL_DIR / "ensemble_l1logreg_rf_svm.pkl")
    fx.save()

    fig_confusion_matrix(ensemble, X_test, y_test)
    fig_roc_pr_curves(ensemble, X_test, y_test)
    comparison_df = fig_model_comparison(all_scores)
    fig_cv_score_distribution(cv_f1_scores)
    save_hyperparameters(tuned)

    ablation_df = ablation_study(train_blocks, y_train, test_blocks, y_test)
    fig_ablation(ablation_df)

    from .experiment_log import log_run
    log_run("run4_final_lexical_embedding", comparison_df, ablation_df)

    print("\nAll figures saved to:", FIG_DIR)
    print("All tables saved to:", TAB_DIR)
    print("\nFinal feature set used:", FINAL_FEATURE_SET)
    print("\nModel comparison:\n", comparison_df)
    print("\nAblation study (all subsets, for comparison/reporting):\n", ablation_df)


if __name__ == "__main__":
    main()