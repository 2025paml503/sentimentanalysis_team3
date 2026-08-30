# Design Decisions and Justification

** Amazon Review Sentiment Classifier - ML Engineering, Flavor C **

This document exists to statisfy one specific expectation from the mini-project
brief(General Expectations):

> *"All design decisions (model choice, drift-detection approach, retraining
> trigger) must be justified in the documentation, not just implementation"*
 
Each section below states the decision, the alternative that were considered
and rejected, and the evidence the decision rests on. Full implementation
detail lives in 'README.md'; this document is the justification, not a repeat
of the code walkthrough.

---

## 1. Model Choice

**Decision:** `LogisticRegression(C=1.0)` on TF-IDF unigrams with class weight balancing 
(`run1_logreg_baseline`), selected from four tracked MLflow experiments.

| Run | Model | n-grams | Vocab | Accuracy | ROC-AUC | F1     |
|---|---|---|---|---|---|--------|
| **run1_logreg_baseline** | **LogisticRegression, C=1.0** | **1** | **5000** | **0.939** | **0.945** | **0.746** |
| run2_logreg_strong_reg | LogisticRegression, C=0.1 | 1 | 5000 | 0.92 | 0.931 | 0.692 |
| run3_logreg_baseline | LogisticRegression, C=1.0 | 1-2 | 20000 | 0.941 | 0.94 | 0.748 |
| run4_naive_bayes | MultinomialNB, α=1.0 | 1-2 | 20000 | 0.881 | 0.887 | 0.024 |

**Selection Criterion:** ROC-AUC (0.945), not accuracy - it is threshold-independent 
and does not flatter a model just because classes happen to be balanced. This is especially 
critical given the 6.5:1 class imbalance in the dataset.

**Class Weight Balancing (Latest Fix):** The most recent training run added `class_weight="balanced"` 
to LogisticRegression models. This penalizes errors on the minority positive class more heavily, 
counteracting the severe class imbalance (2,767 negative vs 420 positive reviews). As a result:
- Positive class recall improved to 0.75 (from previous runs with poor minority class detection)
- ROC-AUC remains strong at 0.945, indicating good discrimination across both classes
- The simpler unigram model (run1) now outperforms the more complex bigram variant (run3) on ROC-AUC

**Why this model won:** With class balancing applied, the unigram baseline achieved the highest 
ROC-AUC (0.945) and better training efficiency. The vocabulary compression from 20K to 5K terms 
also reduces serving latency and memory footprint without sacrificing ROC-AUC performance.

## 2. Drift-Detection Approach

**Decision:** compare production `oov_rate` (share of words in a request not
present in the training vocabulary), `n_words`, and `n_chars` against the 
training baseline using a normalized-shift statistic (`|prod_mean - train_mean|
/ train_std`), flagging any feature that moves than **1 standard deviation **.

**Why `oov_rate` is the primary signal for a text model.** TF-IDF fixes its 
vocabulary at training time; language does not stay fixed. A tabular model's
drift shows up in feature distributions the model was already told about. A
text model's drift shows up as *words the model has never seen at all* - and
`oov_rate` is the direct measurement of that. It is also cheap to compute at 
request time (one vocabulary lookup per word) and cheap to explain to a 
non-ML stakeholder: "this fraction of every review is invisible to the model."

---

## 3. Retraining Trigger Design

**Decision:** an evidence ladder, not a single automatic trigger. Five
independent signals are checked; the *count* of signals triggered decides
the recommendation, and even the strongest recommendation ("RETRAIN") is a 
proposal for a human to approve, not an automated redeploy.

**Why a ladder instead of one threshold.** A single metric crossing a single
threshold is cheap to game (by adjusting the threshold) and cheap to be wrong
about (any one metric can move for a benign reason - a marketing campaign
attracting a different customer segment is not model failure). Requiring
several *independent* signals to agree, where each has its own mechanistic
explanation, is a materially stronger claim than any one of them alone.

**Retraining procedure, once approved** (unchanged from the frozen
deployment path, so nothing about it is improvised under pressure)

The new run is never auto-promoted on a better ROC-AOC alone; the promotion
decision and its justification are written down before `model_store/` is
overwritten - the same discipline this document itself is an instace of.













