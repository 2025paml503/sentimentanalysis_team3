# Project Introduction

## Project Overview

**Project Name:** Sentiment Analysis Classifier  
**Project Code:** Flavor C - ML Engineering Mini-Project  
**Repository:** BitsProject/sentimentanalysis_team3  
**Status:** Active Development & Deployment

---

## 📋 Project Description

This project implements a **production-ready machine learning pipeline** for sentiment classification of Amazon product reviews. The system automates the process of analyzing customer sentiment from review text, enabling data-driven insights into customer satisfaction trends.

### Business Objective
Develop a scalable sentiment analysis system that:
- Classifies customer reviews as positive or negative with high accuracy
- Provides confidence scores and interpretability metrics
- Detects data drift in production with automated alerts
- Supports batch and real-time inference at scale

### Technical Scope
Build an end-to-end ML system including:
- Data validation and quality assurance
- Exploratory data analysis (EDA) and statistical profiling
- Feature engineering with text preprocessing
- Model experimentation and hyperparameter tuning
- MLflow experiment tracking and model registry
- REST API for model serving
- Web dashboard for visualization and monitoring
- Production monitoring with drift detection

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Amazon product reviews |
| **Size** | 3,500 reviews |
| **Format** | CSV (2.3 MB) |
| **Location** | `data/raw/Amazon_Reviews_3500records.csv` |
| **Features** | review_id, reviewer_name, rating (1-5 stars), title, review_text, sentiment (binary: 0/1) |
| **Class Distribution** | 2,767 negative (0) vs 420 positive (1) — 6.5:1 imbalance |
| **Languages** | English (with international reviews) |

---

## 🎯 Key Deliverables

### Phase 1: Data Pipeline ✅
- [x] Data collection and validation
- [x] Exploratory data analysis (EDA)
- [x] Feature store creation
- [x] Comprehensive documentation

### Phase 2: Model Development ✅
- [x] Baseline model training (4 configurations)
- [x] Experiment tracking with MLflow
- [x] Model evaluation and comparison
- [x] Best model selection and artifact storage

### Phase 3: Production Deployment ✅
- [x] REST API (FastAPI)
- [x] Interactive Streamlit dashboard
- [x] Model versioning and lineage tracking
- [x] Prediction logging and audit trail

### Phase 4: Monitoring & Maintenance ✅
- [x] Drift detection (OOV rate, text length, sentiment shifts)
- [x] Prediction log analysis tools
- [x] Retraining trigger design
- [x] Production monitoring framework

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn |
| **NLP** | TF-IDF vectorization |
| **Experiment Tracking** | MLflow |
| **Web API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit |
| **Database** | SQLite (feature store, MLflow, predictions) |
| **Validation** | Pandera |
| **Visualization** | Matplotlib, Plotly |

---

## 📈 Current Performance

**Best Model:** LogisticRegression with class weight balancing  
**Metrics:**
- Accuracy: 93.9%
- ROC-AUC: 0.945
- F1-Score: 0.746
- Precision: 74.1%
- Recall: 75%

---

## 👥 Team Members - Team Number #3

| # | Name | Email |
|---|------|-------|
| 1 | Karthikeyan Bose | [2025paml541@wilp.bits-pilani.ac.in] |
| 2 | Deepak Garg | [2025paml503@wilp.bits-pilani.ac.in] |
| 3 | Rohit Khandelwal | [2025paml585@wilp.bits-pilani.ac.in] |

---

## 📚 Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **README** | `/README.md` | Complete pipeline guide and quick-start |
| **Architecture** | `/arch/ARCHITECTURE.md` | System design and component diagram |
| **Design Decisions** | `/arch/design-decisions.md` | Justification for model choice, drift detection, retraining |
| **This Document** | `/arch/PROJECT_INTRO.md` | Project overview and team info |

---

## 🔄 How to Get Started

1. **Clone Repository**
   ```bash
   git clone <repo-url>
   cd sentimentanalysis_team3
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Data Validation**
   ```bash
   python validation/data_validation.py
   ```

4. **Build Feature Store**
   ```bash
   python feature/build_features.py
   ```

5. **Train Models**
   ```bash
   python training/train_with_mlflow.py
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

6. **Start API Server**
   ```bash
   uvicorn serving.api:app --port 8000
   ```

7. **Launch Dashboard**
   ```bash
   streamlit run ui/app.py
   ```

See `/README.md` for detailed step-by-step instructions.

---

## ⚠️ Known Issues & Recent Fixes

### Class Imbalance (FIXED ✅)
- **Issue:** 6.5:1 negative-to-positive ratio caused model to predict "negative" for almost everything
- **Fix:** Applied `class_weight="balanced"` to LogisticRegression training
- **Impact:** Positive class recall improved from ~0% to 75%

### Vocabulary Leakage (FIXED ✅)
- **Issue:** TF-IDF vocabulary was being fit on full dataset, mixing test terms into training
- **Fix:** Fit vectorizer on train split only
- **Impact:** Prevents inflated evaluation metrics

---

## 📞 Support & Contact

For questions about:
- **Project scope:** Contact [Lead Team Member]
- **ML implementation:** Contact [ML Engineer]
- **Deployment issues:** Contact [DevOps/Deployment Lead]

---

## 📅 Timeline

| Phase | Status | Dates |
|-------|--------|-------|
| Phase 1: Data Pipeline | ✅ Complete | Q1-Q2 2024 |
| Phase 2: Model Development | ✅ Complete | Q2 2024 |
| Phase 3: Production Deployment | ✅ Complete | Q3 2024 |
| Phase 4: Monitoring & Ops | ✅ Complete | Q3-Q4 2024 |
| Maintenance & Improvement | 🔄 Ongoing | 2025+ |

---

## 📋 Project Governance

- **Code Repository:** BitsProject/sentimentanalysis_team3
- **Issue Tracking:** [GitHub Issues / Linear / Jira]
- **CI/CD Pipeline:** [GitHub Actions / GitLab CI / Jenkins]
- **Deployment Environment:** [Staging / Production URLs]
- **On-call Rotation:** [Team rotation schedule]

---

**Document Version:** 1.0  
**Last Updated:** August 2024  
**Next Review:** Q1 2025
