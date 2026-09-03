# AI Customer Support Ticket Classifier - Intent Detection & Ticket Routing with NLP

## 1. Project Title

**AI Customer Support Ticket Classifier** — an interpretable NLP prototype that predicts the
intent of an incoming customer-support message and supports automatic ticket routing.

## 2. Business Problem

A financial-services company receives a large volume of text-based customer-support requests
every day. Support agents currently read each message manually and route it to the correct
support category, which is slow and does not scale. This project builds a small, interpretable
prototype that:

- Predicts the intent behind a new customer message.
- Reports a confidence score for that prediction.
- Decides when a prediction is reliable enough to auto-route, versus when it should be
  handed off to a human agent.

This is **not** a chatbot or a RAG system. The goal is text classification, evaluation, error
analysis, and business-aware routing — not generating conversational responses.

## 3. Dataset

- **Source:** [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI), a public
  benchmark for fine-grained online-banking intent detection.
- **Size (full dataset):** 13,083 customer-service queries across 77 intents
  (10,003 train / 3,080 test examples).
- **License:** CC BY 4.0.
- **Paper:** Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020),
  https://arxiv.org/abs/2003.04807

### Important dataset detail

Loading this dataset through the Hugging Face `datasets` library
(`datasets.load_dataset("PolyAI/banking77")`) is currently unreliable: recent versions of
`datasets` refuse to load it because of an old, unused Python loading script still present in
that repository, and the Hub's usual Parquet fallback is not available for this dataset either.

To avoid this, the script reads the original CSV files directly from the GitHub repository the
BANKING77 paper itself publishes the data in:

```python
train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)
```

This only needs `pandas` (already a dependency) — no `datasets` or `huggingface_hub` package
required. The CSV files already contain readable intent names (e.g. `"card_arrival"`) in a
column called `category`, which the script renames to `label` to match the rest of the code.

### Selected intents (subset of 8, out of the original 77)

| Intent |
|---|
| card_arrival |
| card_not_working |
| cash_withdrawal_not_recognised |
| declined_card_payment |
| lost_or_stolen_card |
| transaction_charged_twice |
| transfer_not_received_by_recipient |
| cash_withdrawal_charge |

These intents were chosen because they represent common, business-relevant banking-support
scenarios, including some that share overlapping vocabulary (e.g. the two `cash_withdrawal_*`
intents), which makes the classification task non-trivial and useful for error analysis.

## 4. Methodology

1. **Data loading** — load BANKING77 directly from its original CSV files on GitHub, and
   filter down to the 8 selected intents (the CSV files already use readable intent names, so
   no id-to-name decoding is needed).
2. **Exploratory Data Analysis (EDA)** — intent frequency, message length (characters and word
   count) by intent, and top TF-IDF terms per intent.
3. **Train/test split** — an 80/20 stratified split (`random_state=42`) is performed **before**
   any text vectorizer is fit, to avoid data leakage.
4. **Keyword baseline** — a simple rule-based classifier using hand-picked keywords per intent,
   used as a sanity-check baseline before building a real model.
5. **Main model** — a scikit-learn `Pipeline` combining `TfidfVectorizer` (unigrams + bigrams,
   English stop words, max 5000 features) with `LogisticRegression`. Fitting the pipeline on
   `X_train` guarantees the TF-IDF vocabulary is learned only from training data.
6. **Evaluation** — per-class precision/recall/F1, macro-F1, and a confusion matrix.
7. **Error analysis** — manual inspection of the misclassified test examples (up to 20, or all
   of them if fewer than 20 occur).
8. **Business-aware routing** — a `predict_intent()` function returns a confidence score, and
   `defer_to_human()` uses a threshold (default `0.6`) to decide whether a ticket should be
   auto-routed or sent to a human agent.

## 5. EDA Findings

*(From an actual run of the script; your own numbers may vary slightly depending on the
train/test split and library versions, but the overall patterns should be similar.)*

- **Intent distribution (Q1):** see `figures/intent_distribution.png`. Message counts are
  fairly balanced across the 8 selected intents (from 122 for `lost_or_stolen_card` up to 217
  for `cash_withdrawal_charge`, out of ~1,500 combined messages). Since no single intent
  dominates the workload, automation benefits would be spread fairly evenly across categories
  rather than concentrated in one bottleneck.
- **Message length by intent (Q2):** see `figures/char_length_by_intent.png` and
  `figures/word_count_by_intent.png`. `lost_or_stolen_card` and `card_not_working` messages
  tend to be the shortest (median ~37-39 characters — short, urgent messages), while
  `cash_withdrawal_not_recognised` and `transfer_not_received_by_recipient` tend to be the
  longest (median ~57-60 characters — customers explaining a sequence of events). This makes
  length a weak but potentially useful secondary signal, not a reliable standalone feature.
- **Top TF-IDF terms (Q3 support):** see `figures/top_tfidf_terms.png`. Each intent has a
  distinctive vocabulary (e.g. "arrive", "new card" for `card_arrival`), which is what both the
  keyword baseline and the TF-IDF model rely on.

## 6. Keyword Baseline (Q3)

A simple rule-based classifier scores each intent by counting keyword matches in the message
and picks the intent with the highest score (`"unknown"` if no keyword matches). On an actual
run, this baseline reached **49.5% accuracy** on the test set — far below the ML model (95.0%,
see below) — which:

- Works reasonably well on messages that contain very explicit phrases.
- Fails when keywords overlap across intents (e.g. "card" appears in several categories) or
  when a message doesn't use any of the predefined keywords — several test errors were labeled
  `"unknown"` because none of the hand-picked keywords appeared in the message at all.
- Motivates the need for a statistical/ML model that can learn weighted, combined evidence
  from the full vocabulary rather than a fixed keyword list.

## 7. TF-IDF + Logistic Regression Model (Q4)

- Vectorizer: `TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)`
- Classifier: `LogisticRegression(max_iter=1000, random_state=42)`
- Wrapped together in a single `Pipeline`, fit only on `X_train`, so there is no data leakage
  between train and test vocabularies.

## 8. Evaluation (Q5)

Results from an actual run on the held-out test set (301 messages):

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| card_arrival | 0.905 | 0.974 | 0.938 |
| card_not_working | 0.931 | 0.900 | 0.915 |
| cash_withdrawal_charge | 0.978 | 1.000 | 0.989 |
| cash_withdrawal_not_recognised | 1.000 | 0.900 | 0.947 |
| declined_card_payment | 0.884 | 0.974 | 0.927 |
| lost_or_stolen_card | 0.952 | 0.833 | 0.889 |
| transaction_charged_twice | 1.000 | 0.953 | 0.976 |
| transfer_not_received_by_recipient | 0.955 | 1.000 | 0.977 |

- **Overall accuracy:** 0.950
- **Macro-F1:** 0.945 (chosen over accuracy alone, since it weighs every intent equally
  regardless of how many test examples it has)
- Confusion matrix — see `figures/confusion_matrix.png`.
- Per-class F1 bar chart — see `figures/f1_per_class.png`.

**Observed pattern:** the single most common confusion pair is `card_not_working` being
predicted as `declined_card_payment` (both describe a card failing at the point of use, so
they share vocabulary like "declined"/"not working"). Separately, `lost_or_stolen_card` has the
lowest recall (0.833) of all 8 intents — several of its messages were confused with
`card_arrival` or `card_not_working` (e.g. "I lost track of my card" reads similarly to a
card-arrival question). This recall gap matters more than the raw confusion count, since it's
also one of the two business-critical intents flagged in Q6 below.

## 9. Error Analysis (Q5 / Section 7 of the code)

The test run produced 15 misclassified examples out of 301 (below the "at least 20" errors the
brief describes inspecting — a larger test set or a stricter model would be needed to reach 20;
all 15 are printed and reviewed below). Recurring patterns:

1. **Overlapping vocabulary between semantically related intents** — e.g. "declined" language
   appears in both `card_not_working` and `declined_card_payment` messages, causing confusion
   in both directions (errors #2, #10, #14).
2. **Very short or ambiguous messages** — e.g. "Help! I can't find my card." (error #6) reads as
   card-not-working just as easily as lost-or-stolen.
3. **Indirect phrasing** — e.g. "Has there been any activity on my card today?" (error #13) is a
   `lost_or_stolen_card` message that doesn't use any obviously urgent language, so it was
   predicted as `card_not_working`.
4. **Double-charge vs. withdrawal-fee overlap** — "Why did I get charged more than once?"
   (error #9) was predicted as `cash_withdrawal_charge` instead of `transaction_charged_twice`,
   since both intents share "charge"-related vocabulary.

## 10. Business Implications (Q6)

Two business-critical intents were singled out for extra scrutiny:

- **`lost_or_stolen_card`** — a **false negative** here (missing a real lost/stolen-card
  report) can lead to real financial loss for the customer, so **recall** is the metric to
  prioritize for this intent. In this run, recall was 0.833 — the lowest of all 8 intents —
  meaning roughly 1 in 6 real lost/stolen-card messages would currently be mis-routed. This is
  a concrete example of where the model's overall high accuracy (0.950) hides a weaker spot on
  exactly the intent where mistakes are most costly.
- **`declined_card_payment`** — both **precision and recall** matter: a false positive can
  confuse or mis-route a customer, while a false negative can delay resolving a real payment
  issue. Recall here was strong (0.974), though precision (0.884) suggests some
  `card_not_working` messages get mis-labeled as `declined_card_payment`.

This is why the project reports per-class metrics instead of a single accuracy number — the
cost of an error is not the same across intents.

## 11. Limitations

- The model is trained on a subset of 8 out of 77 BANKING77 intents; it is not evaluated on
  out-of-scope messages beyond a simple "unknown"/low-confidence fallback.
- TF-IDF + Logistic Regression captures lexical patterns but not deeper semantics — messages
  that describe the same problem with very different wording may be missed.
- The `0.6` confidence threshold is a simple heuristic tuned by inspection, not a formally
  calibrated probability; some high-confidence predictions can still be wrong, and some
  low-confidence ones can still be correct.
- This is a **prototype**, not a production-ready system: there is no monitoring, retraining
  pipeline, or handling of concept drift over time.

## 12. Repository Structure

```
customer-support-ticket-classifier/
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb        # or the .py script below, converted to a notebook
├── src/
│   └── predict.py            # predict_intent() / defer_to_human() as reusable functions
├── figures/                  # saved plots (created automatically by the script)
└── data/
    └── README.md             # notes on the dataset (BANKING77 is downloaded straight
                               # from GitHub by the script, not stored in the repo)
```

## 13. How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script (or the equivalent notebook):
   ```bash
   python ticket_classifier.py
   ```
   The script will:
   - Download the BANKING77 CSV files directly from GitHub with `pandas`
     (requires internet access).
   - Filter the data down to the 8 selected intents.
   - Generate all 5+ required figures into `figures/` (created automatically).
   - Print the keyword baseline accuracy, classification report, macro-F1, and the
     misclassified test examples to the console.
   - Demonstrate `predict_intent()` and `defer_to_human()` on a few example messages.

## 14. Not Included (Out of Scope for This Project)

Per the project brief, this prototype intentionally does **not** use BERT, Transformers, RAG,
LangChain, vector databases, or multi-agent systems — the focus is on a classical, interpretable
NLP baseline.
