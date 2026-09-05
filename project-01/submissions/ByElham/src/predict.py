"""
AI Customer Support Ticket Classifier -- src/predict.py
========================================================

This single file contains the COMPLETE project workflow so the pipeline can be
reproduced from start to finish in one clearly sectioned module:

    [1] Setup and constants
    [2] Data loading (BANKING77 train/test CSVs + 8-intent subset)
    [3] Text preprocessing (self-contained, 7 stages)
    [4] Model training (TF-IDF + Logistic Regression + GridSearchCV)
    [5] Model evaluation on the held-out test set
    [6] Error analysis on the test set
    [7] Saving the fitted model and metadata to models/
    [8] Reusable prediction with confidence and a human-review rule

There are no runtime dependencies on other source files in this project:
the preprocessing implementation in [3] is fully embedded, so this module is
the only code file needed to reproduce or serve the model.

How to use this file
--------------------
As a script, run the full workflow end-to-end (train -> evaluate -> save ->
predict):
    python src/predict.py

As a module, import the pieces you need (used by the notebook and app.py):
    from predict import predict_intent, preprocess_text, load_model, load_metadata
    from predict import load_or_train_model  # lazy train-on-first-use for app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# ===========================================================================
# [1] SETUP AND CONSTANTS
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "ticket_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Fixed seed for full reproducibility of every random step
RANDOM_SEED = 42

# The 8 intents recommended by the project specification
SELECTED_INTENTS = [
    "card_arrival",
    "card_not_working",
    "cash_withdrawal_not_recognised",
    "declined_card_payment",
    "lost_or_stolen_card",
    "transaction_charged_twice",
    "transfer_not_received_by_recipient",
    "cash_withdrawal_charge",
]

# Intents where misrouting is most costly for the business
BUSINESS_CRITICAL_INTENTS = [
    "lost_or_stolen_card",
    "declined_card_payment",
    "transaction_charged_twice",
]

# Human-review rule: defer when top-class confidence is below 0.5.
# With 8 classes, a confidence below 0.5 means the winning class holds less
# than half of the probability mass: several other intents are still plausible,
# so the prediction is genuinely uncertain and should be reviewed by an agent.
CONFIDENCE_THRESHOLD = 0.5

# Hyperparameter grid searched with cross-validation (see [4])
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_df": [0.90, 0.95, 1.0],
    "tfidf__sublinear_tf": [False, True],
    "tfidf__stop_words": [None, "english"],
    "clf__C": [0.5, 1.0, 5.0],
}


# ===========================================================================
# [2] DATA LOADING
#    Loads the bank-support CSV files and keeps only the 8 selected intents.
#    The official BANKING77 train/test split is preserved (filter, not re-split).
# ===========================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter the BANKING77 train/test CSVs to the 8-intent subset.

    Returns
    -------
    (train, test) : tuple of pandas.DataFrame, each with columns
        text (str) and intent (str).
    """
    train = pd.read_csv(DATA_DIR / "train.csv").rename(columns={"category": "intent"})
    test = pd.read_csv(DATA_DIR / "test.csv").rename(columns={"category": "intent"})

    train = train[train["intent"].isin(SELECTED_INTENTS)].reset_index(drop=True)
    test = test[test["intent"].isin(SELECTED_INTENTS)].reset_index(drop=True)

    return train, test


# ===========================================================================
# [3] TEXT PREPROCESSING (self-contained)
#    A 7-stage embedded pipeline applied to every message. Keeping it inside
#    this file guarantees the exact same transformation is used during
#    training and prediction (no train/serve drift, no extra source files):
#      1 lowercase -> 2 whitespace -> 3 punctuation -> 4 tokenize
#      -> 5 contraction expand -> 6 stopwords -> 7 lemmatize
# ===========================================================================
# Compiled resources used by the pipeline
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")

# Common informal contractions found in customer support messages.
# Applied per-token after punctuation removal, so apostrophe-based forms
# (e.g., "n't") are already split away.
CONTRACTIONS: dict[str, str] = {
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "wont": "will not",
    "cant": "cannot",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "mustnt": "must not",
    "neednt": "need not",
    "lets": "let us",
    "thats": "that is",
    "whats": "what is",
    "heres": "here is",
    "theres": "there is",
    "whens": "when is",
    "wheres": "where is",
    "hows": "how is",
    "im": "i am",
    "ive": "i have",
    "ill": "i will",
    "id": "i would",
    "youre": "you are",
    "theyre": "they are",
    "weve": "we have",
    "youve": "you have",
    "pls": "please",
    "plz": "please",
    "bc": "because",
    "u": "you",
    "r": "are",
    "ur": "your",
    "crd": "card",
    "atm": "atm",
    "tv": "television",
}

# English stopwords (standard list, embedded so no external data is needed)
STOPWORDS: set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "also", "like", "really", "much", "even", "still", "well", "back",
    "get", "got", "going", "go", "make", "made", "know", "think", "see",
    "come", "take", "want", "give", "use", "find", "tell", "ask", "work",
    "seem", "feel", "try", "leave", "call", "need", "would", "could",
    "should", "may", "might", "shall", "thing", "things", "way", "time",
}

# Rule-based lemmatizer: maps common word forms to their base. For words not
# in the map, a conservative suffix-stripping heuristic is applied.
LEMMA_MAP: dict[str, str] = {
    # Common verb forms
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "being": "be", "am": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "doing": "do",
    "goes": "go", "went": "go", "gone": "go", "going": "go",
    "gets": "get", "got": "get", "getting": "get",
    "makes": "make", "made": "make", "making": "make",
    "takes": "take", "took": "take", "taken": "take", "taking": "take",
    "comes": "come", "came": "come", "coming": "come",
    "gives": "give", "gave": "give", "given": "give", "giving": "give",
    "finds": "find", "found": "find", "finding": "find",
    "tells": "tell", "told": "tell", "telling": "tell",
    "asks": "ask", "asked": "ask", "asking": "ask",
    "works": "work", "worked": "work", "working": "work",
    "seems": "seem", "seemed": "seem", "seeming": "seem",
    "feels": "feel", "felt": "feel", "feeling": "feel",
    "tries": "try", "tried": "try", "trying": "try",
    "leaves": "leave", "left": "leave", "leaving": "leave",
    "calls": "call", "called": "call", "calling": "call",
    "needs": "need", "needed": "need", "needing": "need",
    "knows": "know", "knew": "know", "known": "know", "knowing": "know",
    "thinks": "think", "thought": "think", "thinking": "think",
    "sees": "see", "saw": "see", "seen": "see", "seeing": "see",
    "charged": "charge", "declined": "decline", "blocked": "block",
    "stolen": "steal", "missing": "miss", "waiting": "wait",
    "arrived": "arrive", "received": "receive", "cancelled": "cancel",
    "canceled": "cancel", "stopped": "stop", "failed": "fail",
    "recognised": "recognise", "recognized": "recognize",
    # Common noun forms
    "cards": "card", "charges": "charge", "payments": "payment",
    "transfers": "transfer", "withdrawals": "withdrawal",
    "transactions": "transaction", "problems": "problem",
    "issues": "issue", "messages": "message", "days": "day",
    "weeks": "week", "months": "month", "years": "year",
    "times": "time", "ways": "way", "things": "thing",
    "requests": "request", "tickets": "ticket", "customers": "customer",
    "accounts": "account", "balances": "balance", "fees": "fee",
    "errors": "error", "amounts": "amount", "dollars": "dollar",
    "purchases": "purchase", "notifications": "notification",
    # Common adjective forms
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    # Banking-specific
    "recognised": "recognise", "recognized": "recognize",
}


def _simple_lemmatize(word: str) -> str:
    """Apply rule-based lemmatization for common English word forms."""
    if word in LEMMA_MAP:
        return LEMMA_MAP[word]
    # Suffix-stripping heuristics (conservative: require stem >= 4 chars)
    if word.endswith("ing") and len(word) > 6:
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]  # running -> runn -> run
        return stem if len(stem) >= 4 else word
    if word.endswith("ed") and len(word) > 5:
        stem = word[:-2]
        return stem if len(stem) >= 4 else word
    if word.endswith("ly") and len(word) > 5:
        stem = word[:-2]
        return stem if len(stem) >= 4 else word
    if word.endswith("es") and len(word) > 5:
        stem = word[:-2]
        return stem if len(stem) >= 4 else word
    if word.endswith("s") and not word.endswith("ss") and len(word) > 5:
        stem = word[:-1]
        return stem if len(stem) >= 4 else word
    return word


def preprocess_text(text: str) -> str:
    """Apply the full 7-stage preprocessing pipeline to a single text string.

    Stages:
        1. Lowercasing
        2. Whitespace normalization
        3. Punctuation removal
        4. Tokenization (whitespace split)
        5. Contraction expansion
        6. Stopword removal
        7. Lemmatization

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Preprocessed text with tokens joined by single spaces.
    """
    # Stage 1: Lowercasing
    text = text.lower()

    # Stage 2: Whitespace normalization
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # Stage 3: Punctuation removal
    text = _PUNCT_RE.sub(" ", text)

    # Stage 4: Tokenization (simple whitespace split)
    tokens = text.split()

    # Stage 5: Contraction expansion (per-token to avoid substring bugs)
    tokens = [CONTRACTIONS.get(t, t) for t in tokens]

    # Stage 6: Stopword removal (drop single-char tokens too)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # Stage 7: Lemmatization
    tokens = [_simple_lemmatize(t) for t in tokens]

    # Rejoin
    return " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Apply preprocess_text to a list of texts."""
    return [preprocess_text(t) for t in texts]


def preprocess_dataframe(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Add a text_clean column to both DataFrames using preprocess_batch.

    Mutates the DataFrames in place so downstream code simply reads
    df["text_clean"]. Rows whose cleaned text is empty are dropped by
    clean_split(), matching the notebook workflow.
    """
    train["text_clean"] = preprocess_batch(train["text"].tolist())
    test["text_clean"] = preprocess_batch(test["text"].tolist())


def clean_split(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return model-ready arrays after dropping fully-stopped rows.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    train = train[train["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    test = test[test["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    return (
        train["text_clean"],
        train["intent"],
        test["text_clean"],
        test["intent"],
    )


# ===========================================================================
# [4] MODEL TRAINING
#    TF-IDF + Logistic Regression chained in a scikit-learn Pipeline so the
#    vectorizer is fitted ONLY on training data (no data leakage). The best
#    hyperparameters are chosen with GridSearchCV + stratified 5-fold CV.
# ===========================================================================
def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline (unfitted).

    lowercase=False because lowercasing is already handled in [3];
    class_weight="balanced" helps the smaller classes.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=False)),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


def train_model(
    X_train: pd.Series, y_train: pd.Series
) -> dict[str, Any]:
    """Run GridSearchCV and return the best fitted pipeline.

    Parameters
    ----------
    X_train : preprocessed training texts (pd.Series of str)
    y_train : training intent labels (pd.Series of str)

    Returns
    -------
    dict with keys: pipeline (best fitted Pipeline), grid_search,
    best_params (dict), cv_macro_f1 (float).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    return {
        "pipeline": grid_search.best_estimator_,
        "grid_search": grid_search,
        "best_params": dict(grid_search.best_params_),
        "cv_macro_f1": float(grid_search.best_score_),
    }


# ===========================================================================
# [5] MODEL EVALUATION
#    Per-class Precision/Recall/F1, Macro-F1 and the confusion matrix are
#    computed on the held-out test set. Accuracy is never reported alone
#    because it hides per-class behaviour.
# ===========================================================================
def evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted pipeline on the test set.

    Returns
    -------
    dict with keys: y_pred (np.ndarray), report (dict from
    sklearn.metrics.classification_report), macro_f1 (float), classes (list),
    cm (np.ndarray).
    """
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    classes = list(pipeline.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    return {
        "y_pred": y_pred,
        "report": report,
        "macro_f1": macro_f1,
        "classes": classes,
        "cm": cm,
    }


# ===========================================================================
# [6] ERROR ANALYSIS
#    Collects every misclassified test example with its predicted class and
#    confidence so failures can be inspected and patterns identified.
# ===========================================================================
def analyze_errors(
    pipeline: Pipeline, test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    """Return a DataFrame containing only the incorrect predictions.

    Columns: text, text_clean, intent, predicted, confidence.
    Confidence = probability assigned to the predicted class.
    """
    errors_df = test.copy()
    errors_df["predicted"] = y_pred
    errors_df["correct"] = errors_df["intent"] == errors_df["predicted"]

    proba_all = pipeline.predict_proba(test["text_clean"])
    classes = list(pipeline.classes_)
    errors_df["confidence"] = [
        proba_all[i, classes.index(p)] for i, p in enumerate(errors_df["predicted"])
    ]
    return errors_df[~errors_df["correct"]].reset_index(drop=True)


# ===========================================================================
# [7] SAVING MODEL AND METADATA
#    The fitted pipeline is serialized with joblib; class names, the confidence
#    threshold and training summary are stored alongside it in JSON.
# ===========================================================================
def save_model(pipeline: Pipeline, metadata: dict) -> None:
    """Persist the fitted pipeline and metadata to models/.

    The models/ directory is created on demand, so simply importing this
    module never leaves stray folders behind.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved to    : {MODEL_PATH}")
    print(f"Metadata saved to : {METADATA_PATH}")


def build_metadata(pipeline: Pipeline, training_summary: dict) -> dict:
    """Assemble the JSON metadata document stored with the model."""
    return {
        "classes": list(pipeline.classes_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "selected_intents": SELECTED_INTENTS,
        "business_critical_intents": BUSINESS_CRITICAL_INTENTS,
        "random_seed": RANDOM_SEED,
        **training_summary,
    }


# ===========================================================================
# [8] PREDICTION
#    load_model / load_metadata read the artifacts saved in [7].
#    predict_intent classifies a single new message with the same preprocessing
#    used during training and applies the human-review rule.
# ===========================================================================
def load_model(model_path: Path | str = MODEL_PATH) -> Any:
    """Load the serialized scikit-learn Pipeline from disk.

    Raises FileNotFoundError if the model has not been saved yet (run the
    workflow first).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run 'python src/predict.py' to train and save the model."
        )
    return joblib.load(path)


def load_metadata(metadata_path: Path | str = METADATA_PATH) -> dict:
    """Load the JSON metadata dict (classes, threshold, summary).

    Raises FileNotFoundError if the metadata file has not been saved yet.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. Run 'python src/predict.py' first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_intent(
    message: str,
    model: Any | None = None,
    metadata: dict | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    """Predict customer intent for a new message.

    Applies the same 7-stage preprocessing used during training before passing
    the text to the TF-IDF + Logistic Regression model.

    Parameters
    ----------
    message : str
        The raw text of a customer support message.
    model : fitted Pipeline, optional
        Pipeline with TfidfVectorizer and LogisticRegression. If None, loaded
        from models/ticket_classifier.joblib.
    metadata : dict, optional
        Dict with 'classes' and 'confidence_threshold'. If None, loaded from
        models/model_metadata.json.
    confidence_threshold : float, optional
        Override the saved threshold for this call.

    Returns
    -------
    dict with keys:
        predicted_intent : str or None
        confidence : float (probability of the top class)
        probabilities : dict[str, float]
        needs_human_review : bool
        review_reason : str or None
    """
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()

    classes = metadata["classes"]
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else metadata.get("confidence_threshold", CONFIDENCE_THRESHOLD)
    )

    # Empty or whitespace-only messages cannot be classified
    if not message.strip():
        return {
            "predicted_intent": None,
            "confidence": 0.0,
            "probabilities": {},
            "needs_human_review": True,
            "review_reason": "Empty message; cannot classify automatically.",
        }

    # Reuse the exact preprocessing pipeline used during training
    text_clean = preprocess_text(message)

    # Preprocessing can remove everything (e.g. message was only stopwords)
    if not text_clean.strip():
        return {
            "predicted_intent": None,
            "confidence": 0.0,
            "probabilities": {},
            "needs_human_review": True,
            "review_reason": "Message contains only stopwords after preprocessing.",
        }

    # Class probabilities from the fitted pipeline
    proba = model.predict_proba([text_clean])[0]
    top_idx = int(np.argmax(proba))
    predicted_intent = classes[top_idx]
    confidence = float(proba[top_idx])

    probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}
    needs_review = confidence < threshold
    review_reason = (
        f"Confidence {confidence:.3f} is below threshold {threshold:.2f}."
        if needs_review
        else None
    )

    return {
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "probabilities": probabilities,
        "needs_human_review": needs_review,
        "review_reason": review_reason,
    }


def load_or_train_model() -> tuple[Any, dict]:
    """Load the trained model and metadata, or train-and-save them on first run.

    This lets ``streamlit run app.py`` work with no prior command: the first
    launch pays the training cost once and caches the artifacts in models/,
    so every subsequent launch loads instantly.
    """
    # Fast path: reuse the saved artifacts when present
    try:
        model = load_model()
        metadata = load_metadata()
        return model, metadata
    except FileNotFoundError:
        pass

    # Cold path: run the full training workflow, then persist the artifacts
    print("Model files missing - running the full training workflow first...")
    train, test = load_data()
    preprocess_dataframe(train, test)
    X_train, y_train, X_test, y_test = clean_split(train, test)
    result = train_model(X_train, y_train)
    eval_result = evaluate_model(result["pipeline"], X_test, y_test)
    metadata = build_metadata(
        result["pipeline"],
        {
            "macro_f1": eval_result["macro_f1"],
            "cv_macro_f1": result["cv_macro_f1"],
            "best_params": {str(k): str(v) for k, v in result["best_params"].items()},
        },
    )
    save_model(result["pipeline"], metadata)
    return result["pipeline"], metadata


# ===========================================================================
# FULL WORKFLOW (script entry point)
#    python src/predict.py
# ===========================================================================
def main() -> None:
    print("=" * 62)
    print("AI Customer Support Ticket Classifier -- full workflow")
    print("=" * 62)

    # [2] Load data
    train, test = load_data()
    print(f"Training examples : {len(train)}")
    print(f"Test examples     : {len(test)}")

    # [3] Preprocess
    preprocess_dataframe(train, test)
    X_train, y_train, X_test, y_test = clean_split(train, test)
    print(f"After cleaning    : {len(X_train)} train / {len(X_test)} test")

    # [4] Train
    n_configs = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"\n[Training] GridSearchCV over {n_configs} configurations...")
    result = train_model(X_train, y_train)
    pipeline = result["pipeline"]
    print("Best CV Macro-F1 : {:.4f}".format(result["cv_macro_f1"]))
    for k, v in result["best_params"].items():
        print(f"  {k}: {v}")

    # [5] Evaluate
    eval_result = evaluate_model(pipeline, X_test, y_test)
    print("\n[Evaluation] Classification report (test set):")
    print(classification_report(y_test, eval_result["y_pred"], digits=3))
    print(f"Macro-F1: {eval_result['macro_f1']:.4f}")

    # [5b] Top confused pairs
    cm = eval_result["cm"]
    classes = eval_result["classes"]
    pairs = [
        (classes[i], classes[j], int(cm[i, j]))
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop confused intent pairs (true -> predicted, count):")
    for true_l, pred_l, cnt in pairs[:6]:
        print(f"  {true_l} -> {pred_l}: {cnt}")

    # [6] Error analysis
    errors = analyze_errors(pipeline, test, y_test, eval_result["y_pred"])
    print(f"\n[Error analysis] {len(errors)} errors on {len(test)} test messages")

    # [7] Save model + metadata
    save_model(
        pipeline,
        build_metadata(
            pipeline,
            {
                "macro_f1": eval_result["macro_f1"],
                "cv_macro_f1": result["cv_macro_f1"],
                "best_params": {str(k): str(v) for k, v in result["best_params"].items()},
            },
        ),
    )

    # [8] Prediction demo
    print("\n[Prediction] Sample predictions with the saved model:")
    examples = [
        "My card was stolen yesterday, please block it immediately.",
        "I was charged twice for the same purchase at the supermarket.",
        "My transfer has not been received by the recipient yet.",
        "I would like to apply for a mortgage.",
    ]
    for msg in examples:
        pred = predict_intent(msg)
        print(
            f"  intent={pred['predicted_intent']:<28} "
            f"confidence={pred['confidence']:.3f} "
            f"review={pred['needs_human_review']}"
        )
    print("\nWorkflow completed.")


if __name__ == "__main__":
    main()