
# AI Customer Support Ticket Classifier - Intent Detection & Ticket Routing with NLP

## Mohammad Amin Nemati 

## Business Problem
The customer-support team of a financial-services company receives many 
text requests every day. Agents currently read each request and manually 
route it to the appropriate support category. The goal of this project is to build 
a small, interpretable NLP prototype that predicts the intent of a new customer message and supports 
automatic ticket routing. 

## Dataset Information
* **Dataset:** BANKING77, a public English-language benchmark for fine-grained online-banking intent classification.
* **Main Dataset Link:** [HuggingFace - BANKING77](https://huggingface.co/datasets/PolyAI/banking77)
* **Original Train CSV:** [PolyAI GitHub Repository](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/train.csv)
* **License:** CC BY 4.0
* **Selected Intents:** `card_arrival`, `card_not_working`, `cash_withdrawal_not_recognised`, 
`declined_card_payment`, `lost_or_stolen_card`, `transaction_charged_twice`, 
`transfer_not_received_by_recipient`, `cash_withdrawal_charge`.

## Methodology
This project focuses on text classification, evaluation, error analysis and business-aware routing. 
The workflow begins with data understanding and text EDA, 
followed by the creation of a simple rule-based keyword baseline. A reproducible machine learning 
pipeline is then built using TF-IDF vectorization (fit strictly on training data to prevent data leakage) 
and Logistic Regression. The model is evaluated using class-level metrics, Macro-F1, and a confusion matrix.

## Repository Structure
```text
customer-support-ticket-classifier/
├── docs/
│   └── StudyBuild_Project01_AI_Ticket_Classifier_Milad_Dates.pdf
├── figures/
│   ├── confusion_matrix.png
│   ├── intent_distribution.png
│   ├── message_structure.png
│   ├── per_class_f1.png
│   └── top_tfidf_terms.png
├── notebooks/
│   └── main.ipynb
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```
## Instructions for Running the Code

1. Clone this repository and open the project directory.
2. Install the environment and dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Open `notebooks/main.ipynb` and run the cells sequentially to reproduce the data download, visualizations, model training, and evaluation metrics.
4. To test the automated routing logic on new text messages, scroll to the final section of the notebook and use the `predict_ticket_intent` function.

## EDA Findings

### Q1: What are customers contacting us about most often?
Transactional and digital routing discrepancies dominate the support queue. 
The top three intents are `cash_withdrawal_charge` (15.0%), 
`transaction_charged_twice` (14.8%), and `transfer_not_received_by_recipient` 
(14.5%). In contrast, physical card logistics like `lost_or_stolen_card` 
represent the lowest volume (6.9%). The operational implication is that the 
bulk of the team's workload revolves around tracing digital money movements 
rather than handling physical hardware issues; prioritizing automation for 
these transactional workflows will yield the highest ROI.

### Q2: Do message length and structure differ by intent?
While most categories share a similar structure with medians hovering between 
10-15 words, `transfer_not_received_by_recipient` shows a noticeably wider 
distribution and a higher median length. Customers likely use more words to 
explain the context of a missing transfer. However, because the boxes (the 
middle 50% of data) overlap so heavily across all intents, message length 
alone remains an insufficient feature for reliable classification. The system 
must rely on natural language processing to extract the specific vocabulary used.

### Q3: Can a simple keyword system route tickets?
A keyword-based router succeeds only when customers use explicit, 
predictable vocabulary (e.g., "lost", "twice", "declined"). However, it fails in two major areas:
1. **Low Coverage / High Unrouted Rate:** Customers frequently use synonyms, 
varied phrasing, or indirect explanations (e.g., "my card was swallowed" or "money didn't show up 
in their account"), leading to tickets falling through as `unknown`.
2. **Lexical Overlap & False Positives:** Hardcoded rules cannot weigh context; 
a message mentioning "charged a fee for card delivery" may falsely trigger withdrawal or arrival 
rules instead of the true intent.

Machine learning is necessary because statistical models (like TF-IDF + Logistic Regression) 
learn soft statistical weights across the entire vocabulary and combinations of n-grams 
rather than relying on brittle, exact string matches.

### Q4: Can classical NLP predict customer intent?
Yes. A reproducible TF-IDF and Logistic Regression pipeline effectively 
predicts customer intents. Fitting the vectorizer strictly on the training 
split prevents data leakage into test sets.

### Q5: Which intents does the model understand well, and which does it confuse?
The classifier achieved a Macro-F1 score of **0.922**.

- **Strongest Intent:** `transaction_charged_twice` (F1: **0.986**), 
  benefiting from distinct, domain-specific vocabulary.
- **Weakest Intent:** `card_not_working` (F1: **0.821**).
- **Top Confusion Pair:** `card_not_working` was misclassified as 
  `declined_card_payment` (4 instances). Linguistically, this 
  occurs due to shared vocabulary patterns across similar transaction workflows.

### Visualizations Interpretation
The generated visualizations confirm that classification success is driven by 
vocabulary rather than message length. The TF-IDF plot demonstrates how the model 
successfully isolates unique, intent-specific n-grams (e.g., "stolen", "fee") 
to establish its routing logic. Furthermore, the per-class F1-score chart visually 
validates the model's balanced performance across both majority and minority classes.

### Q6: Are all errors equally costly?
No. Errors in security-related intents like `lost_or_stolen_card` and 
`cash_withdrawal_not_recognised` carry a significantly higher business 
cost than logistical inquiries like `card_arrival`.

If a stolen card report is misclassified as a general inquiry (a False 
Negative), the ticket may be routed to a low-priority 48-hour queue. 
This delay exposes the customer to continued fraud and the bank to severe 
financial liability and reputational damage.

Therefore, for these business-critical intents, **Recall** is the most 
important metric. The model must maximize Recall to ensure every 
potential security threat is caught and escalated immediately, even if 
it results in lower Precision (i.e., routing a few harmless tickets to 
the urgent fraud team as false alarms).

### Error Analysis
A manual inspection of 20 incorrect predictions reveals several linguistic 
challenges for the TF-IDF baseline:

1. **Ambiguous Phrasing:** Customers frequently use indirect language (e.g., 
   'my card got swallowed by the machine') which lacks the explicit keywords 
   associated with the `lost_or_stolen_card` intent.
2. **Multi-Intent Messages:** Tickets detailing a complex timeline (e.g., 
   'I lost my card, ordered a new one, but they charged me twice for delivery') 
   confuse the classifier, which heavily weights the last visible keyword.
3. **Missing Context:** Short queries (e.g., 'where is it?') do not contain 
   enough substantive n-grams for the model to confidently map them to 
   `card_arrival` versus a missing transfer.

### Q7: How should the system handle a new message?
New messages are passed to a reusable Python function that returns both 
the predicted intent and an interpretable confidence score. By utilizing 
Scikit-Learn's `predict_proba` method, the system extracts the probability 
of the assigned class. This provides necessary context for automated routing 
workflows, allowing the system to measure its own certainty rather than 
relying on a blind guess.

### Q8: When should the system defer to a human?
The system defers to human triage when the model's top predicted class 
probability falls below an empirical confidence threshold (default: 
`0.50`). On the test set, this rule flags approximately 
**29.1%** of queries for manual human review.

**Justification:** In an 8-class system where a random guess yields 
12.5% confidence, a probability below 50% indicates significant 
distribution entropy and ambiguity between two or more intents. 
Routing low-confidence tickets prevents severe misclassification.

**Limitations:**
- **Overconfident Misclassifications:** A linear model can assign high 
  probability to out-of-distribution text if strong isolated n-grams 
  are present, bypassing the human fallback rule.
- **Capacity Trade-Off:** Lowering the threshold risks routing errors, 
  while raising it increases agent workload and operational cost.

### Q8: When should the system defer to a human?
The system defers to human triage when the model's top predicted class 
probability falls below an empirical confidence threshold (default: 
`0.50`). On the test set, this rule flags approximately 
**29.1%** of queries for manual human review.

**Justification:** In an 8-class system where a random guess yields 
12.5% confidence, a probability below 50% indicates significant 
distribution entropy and ambiguity between two or more intents. 
Routing low-confidence tickets prevents severe misclassification.

**Limitations:**
- **Overconfident Misclassifications:** A linear model can assign high 
  probability to out-of-distribution text if strong isolated n-grams 
  are present, bypassing the human fallback rule.
- **Capacity Trade-Off:** Lowering the threshold risks routing errors, 
  while raising it increases agent workload and operational cost.

### Production Readiness Disclaimer
**Please note:** This model is a proof-of-concept prototype trained on 
an 8-intent subset of the BANKING77 dataset to demonstrate baseline NLP 
routing capabilities. It is **not production-ready**. Deploying this to 
a live customer-facing environment would require scaling the model to the 
full 77-class dataset, implementing strict data privacy (PII) scrubbers, 
and wrapping the pipeline in a robust, containerized API (e.g., FastAPI/Docker).
