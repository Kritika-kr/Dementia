"""
Trains and tunes each model with StandardScaler + SMOTE applied only
INSIDE each CV training fold (via imblearn.pipeline.Pipeline), so no
resampled or scaled information ever leaks into a validation fold.

Run after data_preparation.py and with feature_extraction.py available.
"""
import joblib
from gensim.models import Word2Vec
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .data_preparation import load_raw_data, speaker_level_split
from .feature_extraction import FeatureExtractor

MODEL_GRID = {
    "logreg": (
        LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED),
        {"clf__C": [0.01, 0.1, 1, 10], "select__k": [100, 200, 300]},
    ),
    "l1_logreg": (
        LogisticRegression(penalty="l1", solver="liblinear",
                            max_iter=2000, random_state=config.RANDOM_SEED),
        {"clf__C": [0.01, 0.1, 1, 10], "select__k": [100, 200, 300]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_SEED),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20],
         "select__k": [100, 200, 300]},
    ),
    "svm_rbf": (
        SVC(kernel="rbf", probability=True, random_state=config.RANDOM_SEED),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"],
         "select__k": [100, 200, 300]},
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_SEED),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6],
         "select__k": [100, 200, 300]},
    ),
}


def build_pipeline(estimator, k_features=300):
    """Scaler -> SelectKBest -> SMOTE -> classifier, all refit per CV fold.

    Feature selection happens before resampling: with ~600+ raw features
    (TF-IDF + Count + embeddings + graph + linguistic) and only ~435 training
    transcripts, this is a high-dimension-low-sample regime where reducing
    to the most informative features first improves generalization rather
    than just fitting training noise.
    """
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
        ("smote", SMOTE(random_state=config.RANDOM_SEED)),
        ("clf", estimator),
    ])


def tune_model(name, X_train, y_train, cv):
    estimator, grid = MODEL_GRID[name]
    pipe = build_pipeline(estimator)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"[{name}] best F1={search.best_score_:.3f}  params={search.best_params_}")
    return search.best_estimator_


def main():
    df = load_raw_data()
    train_df, test_df = speaker_level_split(df)

    # Word2Vec trained on the TRAIN transcripts only
    tokenized = [t.split() for t in train_df["text"]]
    w2v = Word2Vec(sentences=tokenized, vector_size=config.WORD2VEC_DIM,
                    window=5, min_count=1, seed=config.RANDOM_SEED)

    fx = FeatureExtractor().fit(train_df["text"], w2v.wv)
    train_blocks = fx.transform(train_df["text"])
    test_blocks = fx.transform(test_df["text"])
    X_train = FeatureExtractor.concat_blocks(train_blocks)
    X_test = FeatureExtractor.concat_blocks(test_blocks)
    y_train, y_test = train_df["label"].values, test_df["label"].values

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)

    tuned = {name: tune_model(name, X_train, y_train, cv) for name in MODEL_GRID}

    # Soft-voting ensemble: L1-LR + RF + SVM, each already tuned above
    ensemble = VotingClassifier(
        estimators=[
            ("l1_logreg", tuned["l1_logreg"]),
            ("random_forest", tuned["random_forest"]),
            ("svm_rbf", tuned["svm_rbf"]),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    joblib.dump(ensemble, config.MODEL_DIR / "ensemble_l1logreg_rf_svm.pkl")
    fx.save()

    # Held-out test performance is computed in evaluate.py using these artifacts.
    return ensemble, X_test, y_test, tuned


if __name__ == "__main__":
    main()