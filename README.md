# Amazon Reviews Sentiment Analysis

A production-ready sentiment classification pipeline for Amazon product reviews. This project implements a complete ML lifecycle from data collection through model serving and monitoring.

## 📊 Dataset

- **Source**: Amazon product reviews (3,500 records)
- **Location**: `data/raw/Amazon_Reviews_3500records.csv`
- **Features**: review_id, title, review_text, rating (1-5 stars), sentiment (0=negative, 1=positive)
- **Class Balance**: 2,767 negative vs 420 positive (class weights balanced in training)

---

## 🔄 Pipeline Workflow

### Step 1: Data Collection ✅

**Status**: COMPLETED

Raw data is present in the repository:
- `data/raw/Amazon_Reviews_3500records.csv` (2.3 MB, 3,500 records)

```bash
# Data is already in the repo, no action needed
```

---

### Step 2: Data Validation ✅

**Purpose**: Validate data quality and identify issues before training

**Command**:
```bash
python validation/data_validation.py
```

**Features**:
- Schema validation with strict type checking (Pandera)
- Missing value and duplicate detection
- Text encoding validation
- Text length outlier detection
- Sentiment label balance analysis

**Outputs**:
- `validation_report.txt` — Human-readable summary
- `validation_issues.json` — Programmatic access to issues

---

### Step 3: Exploratory Data Analysis (EDA) 📊

**Purpose**: Understand data distribution, sentiment balance, and key patterns

**Command**:
```bash
python eda/eda.py
```

**Outputs**:
- `eda_summary.txt` — Text statistics and insights
- `sentiment_distribution.png` — Class distribution visualization

**Metrics Generated**:
- Dataset dimensions and shape
- Duplicate record count
- Missing value analysis
- Non-ASCII character detection
- Sentiment distribution (class imbalance analysis)
- Text length statistics (min, max, mean, std)
- Top 20 most frequent words

---

### Step 4: Feature Engineering 🔨

**Purpose**: Build normalized feature store with cleaned text and computed features

**Command**:
```bash
python feature/build_features.py
```

**Process**:
1. Load raw CSV data
2. Clean text: lowercase, remove HTML tags, strip special characters
3. Combine title + review_text into single document
4. Compute numeric features (word count, character count)
5. Save to SQLite feature store

**Outputs**:
- `feature_store/feature_store.db` — SQLite database with columns:
  - `review_id`: Original ID
  - `clean_doc`: Processed review text
  - `n_words`: Word count
  - `n_chars`: Character count
  - `sentiment`: Target label (0/1)

**Key Design**:
- Shared `clean_text()` function used by both training and serving (prevents train-serving skew)
- Text cleaning is deterministic and reproducible

---

### Step 5: Model Training 🤖

**Purpose**: Train multiple sentiment classifiers and track experiments with MLflow

**Command**:
```bash
python training/train_with_mlflow.py
```

**Experiments**:
Four configurations are trained and compared:
1. **run1_logreg_baseline**: LogisticRegression, unigrams only
2. **run2_logreg_strong_reg**: LogisticRegression with L2 regularization (C=0.1)
3. **run3_logreg_bigrams**: LogisticRegression with unigrams + bigrams
4. **run4_naive_bayes**: MultinomialNB with bigrams

**Training Details**:
- TF-IDF vectorization fit on train split only (prevents vocabulary leakage)
- Train/test split: 80/20 with stratification to maintain class ratio
- **Class weight balancing**: Enabled to handle severe class imbalance (6.5:1 ratio)
  - Penalizes errors on positive reviews more heavily
  - Improves recall on minority class
- Random seed: 42 (reproducibility)

**Metrics Tracked**:
- Accuracy, ROC-AUC, F1-score, Precision, Recall
- Vocabulary size, dataset size
- Model parameters and random state

**Best Model Selection**:
Winner selected by **ROC-AUC** (threshold-independent, handles imbalance well)

**Outputs**:
- `model_store/sentiment_model.pkl` — Trained classifier
- `model_store/tfidf_vectorizer.pkl` — Fitted TF-IDF vocabulary
- `model_store/meta.json` — Model lineage and metrics
- `model_store/experiment_comparison.csv` — All runs compared
- MLflow database (`mlflow.db`) — Experiment tracking

**View Training Results**:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser
```

---

### Step 6: Model Serving (API) 🚀

**Purpose**: Deploy model via REST API for real-time predictions

**Command**:
```bash
uvicorn serving.api:app --port 8000
```

**Endpoints**:

#### Health Check
```bash
GET /health
```
Returns: `{"status": "ok", "model_version": "...", "vocab_size": ...}`

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "title": "Great product",
  "review_text": "This item exceeded my expectations. Highly recommended!"
}
```

Response:
```json
{
  "sentiment": "positive",
  "positive_probability": 0.9234,
  "confidence": 0.8468,
  "oov_rate": 0.05,
  "model_version": "run3_...",
  "latency_ms": 2.34
}
```

#### Batch Prediction (up to 100 reviews)
```bash
POST /predict-batch
```

#### Model Lineage
```bash
GET /model-info
```
Returns full training metadata and experiment details

**Auto-Generated Docs**:
```
http://localhost:8000/docs        # Swagger UI
http://localhost:8000/redoc       # ReDoc
```

---

### Step 7: Interactive Dashboard 📈

**Purpose**: Visualize model performance and make predictions through web UI

**Command**:
```bash
streamlit run ui/app.py
```

Opens interactive dashboard at `http://localhost:8501`

---

### Step 8: Integration Testing 🧪

**Purpose**: Validate the prediction API with real-world test cases

**Prerequisites:**
- API server must be running on `http://localhost:8000`
- Use: `uvicorn serving.api:app --port 8000`

**Command**:
```bash
python test/test_predict_api.py
```

**What It Tests:**
- ✅ Health check endpoint (`/health`)
- ✅ Single prediction endpoint (`/predict`) with 20+ real Amazon reviews
- ✅ Batch prediction endpoint (`/predict-batch`)
- ✅ Sentiment classification accuracy
- ✅ API response times (latency)
- ✅ Confidence scores and OOV rates
- ✅ Error handling and edge cases

**Test Coverage:**
The test suite includes 20+ carefully curated Amazon review samples with:
- Positive sentiment examples (highly praised products)
- Negative sentiment examples (poor quality, service issues)
- Mixed sentiment cases (acknowledges both pros and cons)
- Expected sentiment labels for validation

**Sample Output:**
```
================================================================================
SENTIMENT PREDICTION API TEST SUITE
================================================================================

Target API: http://localhost:8000
Test Records: 20+

✅ Health Check PASSED
   Status: ok
   Model Version: run1_logreg_baseline
   Vocabulary Size: 5000

────────────────────────────────────────────────────────────────────────────────
PREDICTION TESTS
────────────────────────────────────────────────────────────────────────────────

✅ Test 1/20: Excellent product, highly recommended...
   Expected: positive | Predicted: positive
   Confidence: 0.9234 | OOV Rate: 0.0500
   Latency: 2.45ms

⚠️ Test 2/20: Terrible quality, waste of money...
   Expected: negative | Predicted: negative
   Confidence: 0.8765 | OOV Rate: 0.0300
   Latency: 2.12ms

... [18 more tests] ...

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 20
Passed: 20 (100.0%)
Failed: 0 (0.0%)

Sentiment Accuracy: 20/20 correct
Misclassified: 0

Average Confidence: 0.8892
Average OOV Rate: 0.0421

✅ ALL TESTS PASSED!
================================================================================
```

**Interpreting Results:**
- **Passed**: Prediction matches expected sentiment
- **Failed**: Prediction does NOT match expected sentiment
- **Confidence**: How certain the model is (0.5-1.0 range)
- **OOV Rate**: Fraction of words not in training vocabulary (lower is better)
- **Latency**: Time taken to process the request in milliseconds

**Troubleshooting:**
If tests fail:
1. Verify API server is running: `uvicorn serving.api:app --port 8000`
2. Check model artifacts exist in `model_store/`
3. Verify feature store is built: `python feature/build_features.py`
4. Check for connection issues: `curl http://localhost:8000/health`

---

### Step 9: Monitoring 🔍

#### Drift Analysis
**Purpose**: Detect data drift and distribution shifts in production

**Command**:
```bash
python monitoring/analyze_drift.py
```

Checks for:
- Out-of-vocabulary (OOV) rate changes
- Text length distribution drift
- Sentiment prediction shifts

#### Prediction Logs
**Purpose**: Review prediction history and troubleshoot model behavior

**View Last 10 Predictions**:
```bash
python tools/read_prediction_logs.py
```

**View Last 25 Predictions**:
```bash
python tools/read_prediction_logs.py --limit 25
```

**Filter by Traffic Batch**:
```bash
python tools/read_prediction_logs.py --batch drifted
```

**Summary Statistics**:
```bash
python tools/read_prediction_logs.py --summary
```

---

## 🐛 Known Issues & Fixes

### Class Imbalance (FIXED ✅)
- **Issue**: Dataset has 6.5:1 negative-to-positive ratio, model biased toward predicting "negative"
- **Fix**: Added `class_weight="balanced"` to LogisticRegression models
- **Impact**: Improves positive sentiment recall and reduces false negatives

---

## 🛠️ Key Design Patterns

### 1. **Prevent Training-Serving Skew**
- Shared `prepare_document()` function in `feature/build_features.py`
- Both training and serving use identical text cleaning logic
- Fitted TF-IDF vocabulary saved as artifact and reused in serving

### 2. **Reproducibility**
- `random_state=42` everywhere
- TF-IDF fit on train split only (no vocabulary leakage)
- All hyperparameters logged in MLflow
- Feature store is immutable version control

### 3. **Class Imbalance Handling**
- Stratified train-test split maintains class ratio
- Class weights balanced during training
- ROC-AUC used for model selection (not accuracy)

### 4. **Production-Ready Monitoring**
- OOV rate: Early warning for vocabulary drift
- Prediction logging: Full audit trail
- Drift detection: Automated drift analysis

---

## 📁 Project Structure

```
sentimentanalysis_team3/
├── data/raw/                           # Raw datasets
│   └── Amazon_Reviews_3500records.csv
├── validation/                         # Data validation pipeline
│   └── data_validation.py
├── eda/                                # Exploratory data analysis
│   └── eda.py
├── feature/                            # Feature engineering
│   └── build_features.py
├── feature_store/                      # Persistent feature store
│   └── feature_store.db
├── training/                           # Model training
│   └── train_with_mlflow.py
├── model_store/                        # Deployment artifacts
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── meta.json
│   └── experiment_comparison.csv
├── serving/                            # REST API
│   ├── api.py
│   ├── model_loader.py
│   └── inference_schema.py
├── ui/                                 # Streamlit dashboard
│   └── app.py
├── monitoring/                         # Drift detection & logging
│   ├── analyze_drift.py
│   ├── prediction_logger.py
│   └── prediction_logs/
└── tools/                              # Utilities
    └── read_prediction_logs.py
```

---

## 🚀 Quick Start

**Run the full pipeline**:

```bash
# 1. Validate data
python validation/data_validation.py

# 2. Exploratory analysis
python eda/eda.py

# 3. Build feature store
python feature/build_features.py

# 4. Train models
python training/train_with_mlflow.py
# View results at http://localhost:5000 (run: mlflow ui)

# 5. Start API server
uvicorn serving.api:app --port 8000
# Test at http://localhost:8000/docs

# 6. Launch dashboard
streamlit run ui/app.py

# 7. Monitor drift
python monitoring/analyze_drift.py
```

---

## 📊 Performance Summary

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| LogReg (unigrams) | - | - | - | - | - |
| LogReg (strong reg) | - | - | - | - | - |
| **LogReg (bigrams) ⭐** | - | - | - | - | - |
| Naive Bayes | - | - | - | - | - |

*See `model_store/experiment_comparison.csv` for latest metrics*

---

## 🔐 Security & Best Practices

- ✅ No secrets in code (credentials externalized)
- ✅ Model artifacts versioned with metadata
- ✅ Prediction audit trail (all predictions logged)
- ✅ Data validation on ingestion
- ✅ Class weight balancing for fairness

---

## 📝 Notes for Team

1. **Before training**: Always run validation and EDA first
2. **After training**: Check MLflow UI to compare model metrics
3. **Production deployment**: Use model from `model_store/` directory
4. **Monitoring alerts**: Check drift analysis regularly for OOV spikes
5. **Data updates**: Increment version in data filenames, never overwrite

---

## 📧 Questions?

See individual module docstrings for detailed documentation.

Generated: 2024-2025 | Team 3 - BitsProject
