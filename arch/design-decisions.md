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

** Decision:** `LogisticRegression(C=1.0)` on TF-IDF unigrams+bigrams
(`run3_logreg_bigrams`), selected from four tracked MLflow experiments.

| Run | Model | n-grams | Vocab | Accuracy | ROC-AUC | F1         |
|---|---|---|---|---|---|------------|
| run1_logreg_baseline | LogisticRegression, C=1.0 | 1 | 261 | 0.828 | 0.9144 | 0.8333     |
| run2_logreg_strong_reg | LogisticRegression, C=0.1 | 1 | 261 | 0.821 | 0.9135 | 0.8254     |
| **run3_logreg_bigrams** | **LogisticRegression, C=1.0** | ** 1-2** | **2675** | **0.937** | **0.9495** | **0.9366** |
| run4_naive_bayes | MultinomialNB, a=1.0 | 1-2 | 2675 | 0.872 | 0.9362 | 0.8752 |

**Selectin Criterion:** ROC-AUC, not accuracy - it is threshold-independent
and does not flatter a model just because the classes happen to be balanced.

**Why bigrams win.** ~30% of the dataset is deliberately mixed-sentiment
review text of the form *"the design is lovely, **but** the battery failed."*
A unigram model see `lovely` and `failed` as independent evidence and lands
near the decision boundary. Bigrams preserve `but battery`, `failed after`, 
`not worth` - the contrast phrases that actually decide the label. That is a 
structural property of review text, so the +0.109 accuracy / +0.035 ROC-AUC
jump over the unigrams baseline is mechanistic, not noise.

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













