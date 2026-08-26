# README file


## Task 1 Status: DATA COLLECTION

✅ COMPLETED — Raw data is already present in the repository:

    data/raw/Amazon_Reviews_3500records.csv (2.3 MB, 3,500 records)
    data/raw/amazon_reviews.txt (58 KB, text format)

The data collection phase has been successfully completed with two data sources ready for validation.


## 🔍 Task 2 Implementation: DATA VALIDATION

created a comprehensive data validation pipeline at validation/data_validation.py with the following features:
Core Components:

    Schema Validation (Pandera)
        CSV schema with strict type checking for review_id, title, review_text, sentiment
        Text data schema for review and label columns
        Automatic type coercion and constraint validation

    Data Quality Checks
        ✅ Missing value detection
        ✅ Duplicate record identification
        ✅ Text encoding validation (ASCII/non-English characters)
        ✅ Text length outlier detection (unusually short/long reviews)
        ✅ Sentiment label balance analysis

    Comprehensive Reporting
        Detailed validation report (validation_report.txt)
        JSON-formatted issues log (validation_issues.json)
        Summary statistics and memory usage
        Actionable recommendations for data cleaning

    Dual Format Support
        Handles both CSV and tab-separated text formats
        Automatic path resolution relative to project root
        Graceful error handling for missing files

How to Run:
bash

cd sentimentanalysis_team3
python validation/data_validation.py

This will generate:

    validation_report.txt — Human-readable validation summary
    validation_issues.json — Programmatic access to issues for downstream processes

Key Features:

    🎯 Pandera schemas ensure reproducible validation logic
    📊 IQR-based outlier detection for text length anomalies
    🔢 Class balance metrics identify potential imbalance issues
    📝 Pre-trained patterns check for non-ASCII encoding problems
    ✨ Production-ready with proper error handling and logging

### Feature store
    python features/build_features.py

### Model training
    python training/train_with_mlflow.py

###MLflow UI command
    mlflow ui --backend-store-uri sqlite:///mlflow.db

###To run an application, use below command
    uvicorn serving.api:app --port 8000

### Streamlit ui
    streamlit run ui/app.py
