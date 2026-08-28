"""
validation/data_validation.py
==============================
Task 2 — Data Validation Pipeline

Validates raw data quality, schemas, and missing values before analysis.
This module ensures data integrity before it enters the feature engineering pipeline.

Key responsibilities:
1. Load raw data from both CSV and text formats
2. Define and validate schema using Pandera
3. Check for missing values, duplicates, and data type mismatches
4. Validate text fields (encoding, length, non-null)
5. Generate comprehensive validation reports
6. Flag problematic records for manual review

Output: validation_report.txt (detailed findings and summary statistics)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check, Index


# ─── Project Paths ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Amazon_Reviews_3500records.csv"
DATA_RAW_TXT = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.txt"
VALIDATION_REPORT = PROJECT_ROOT / "validation_report.txt"
VALIDATION_ISSUES = PROJECT_ROOT / "validation_issues.json"


# ─── Schema Definitions ─────────────────────────────────────
# Pandera schemas define expected structure and constraints

CSV_SCHEMA = DataFrameSchema(
    {
        "review_id": Column(
            dtype="object",  # Can be string or int
            checks=[
                Check(lambda s: s.notna().all(), error="review_id cannot be null"),
                # Cast to string first to safely use string accessor (handles ints)
                Check(lambda s: s.astype(str).str.len() > 0, error="review_id must not be empty"),
            ],
            required=True,
        ),
        "title": Column(
            dtype="object",
            checks=[
                Check(lambda s: s.str.len() >= 0, error="title length must be >= 0"),
                # Title can be empty, but if present should be reasonable length
                Check(lambda s: s.str.len() <= 500, error="title too long (> 500 chars)"),
            ],
            nullable=True,  # Can be null
            required=True,
        ),
        "review_text": Column(
            dtype="object",
            checks=[
                Check(lambda s: s.str.len() > 0, error="review_text cannot be empty"),
                Check(lambda s: s.str.len() <= 10000, error="review_text too long (> 10000 chars)"),
            ],
            required=True,
        ),
        "sentiment": Column(
            dtype="int64",
            checks=[
                Check(lambda s: s.isin([0, 1]), error="sentiment must be 0 (negative) or 1 (positive)"),
                Check(lambda s: s.notna().all(), error="sentiment cannot be null"),
            ],
            required=True,
        ),
    },
    strict=False,  # Allow extra columns (for future additions)
    coerce=True,  # Try to coerce types
)


TXT_SCHEMA = DataFrameSchema(
    {
        "review": Column(
            dtype="object",
            checks=[
                Check(lambda s: s.str.len() > 0, error="review text cannot be empty"),
                Check(lambda s: s.str.len() <= 10000, error="review too long (> 10000 chars)"),
            ],
            required=True,
        ),
        "label": Column(
            dtype="int64",
            checks=[
                Check(lambda s: s.isin([0, 1]), error="label must be 0 or 1"),
                Check(lambda s: s.notna().all(), error="label cannot be null"),
            ],
            required=True,
        ),
    },
    strict=True,  # Exactly these columns
    coerce=True,
)


# ─── Data Loading Functions ─────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV data with proper type handling."""
    df = pd.read_csv(path, keep_default_na=False)
    return df


def load_txt(path: Path) -> pd.DataFrame:
    """Load tab-separated text data."""
    df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"], encoding="utf-8")
    return df


# ─── Data Quality Checks ─────────────────────────────────────

class DataValidator:
    """Orchestrates all data validation checks and report generation."""

    def __init__(self):
        self.issues = {
            "schema_errors": [],
            "missing_values": {},
            "duplicates": 0,
            "encoding_issues": [],
            "outliers": [],
            "warnings": [],
        }
        self.summary = {}

    def validate_csv(self, df: pd.DataFrame) -> bool:
        """Validate CSV data against schema."""
        print("🔍 Validating CSV schema...")
        try:
            CSV_SCHEMA.validate(df, lazy=True)
            print("✅ CSV schema validation passed")
            return True
        except pa.errors.SchemaErrors as err:
            print(f"❌ Schema validation failed: {len(err.schema_errors)} errors")
            for e in err.schema_errors:
                self.issues["schema_errors"].append(str(e))
            return False

    def validate_txt(self, df: pd.DataFrame) -> bool:
        """Validate text data against schema."""
        print("🔍 Validating text schema...")
        try:
            TXT_SCHEMA.validate(df, lazy=True)
            print("✅ Text schema validation passed")
            return True
        except pa.errors.SchemaErrors as err:
            print(f"❌ Schema validation failed: {len(err.schema_errors)} errors")
            for e in err.schema_errors:
                self.issues["schema_errors"].append(str(e))
            return False

    def check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in critical columns."""
        print("🔍 Checking missing values...")
        missing = df.isnull().sum()
        if missing.any():
            self.issues["missing_values"] = missing[missing > 0].to_dict()
            print(f"⚠️  Found missing values in {len(self.issues['missing_values'])} columns")
        else:
            print("✅ No missing values found")

    def check_duplicates(self, df: pd.DataFrame, subset: list[str]) -> None:
        """Check for duplicate rows."""
        print("🔍 Checking for duplicates...")
        n_duplicates = df.duplicated(subset=subset).sum()
        self.issues["duplicates"] = int(n_duplicates)
        if n_duplicates > 0:
            print(f"⚠️  Found {n_duplicates} duplicate records")
        else:
            print("✅ No duplicates found")

    def check_encoding(self, df: pd.DataFrame, text_col: str) -> None:
        """Check for non-ASCII/non-English characters."""
        print("🔍 Checking text encoding...")
        non_ascii = df[text_col].apply(
            lambda text: bool(re.search(r"[^\x00-\x7F]", str(text)))
        ).sum()
        if non_ascii > 0:
            indices = df[df[text_col].apply(
                lambda text: bool(re.search(r"[^\x00-\x7F]", str(text)))
            )].index.tolist()[:5]  # First 5 examples
            self.issues["encoding_issues"].append(
                f"{non_ascii} records with non-ASCII characters (examples at indices: {indices})"
            )
            print(f"⚠️  Found {non_ascii} records with non-ASCII characters")
        else:
            print("✅ All text is ASCII-compatible")

    def check_text_length_outliers(self, df: pd.DataFrame, text_col: str) -> None:
        """Identify unusually short or long text."""
        print("🔍 Checking text length distribution...")
        lengths = df[text_col].str.len()
        
        q1 = lengths.quantile(0.25)
        q3 = lengths.quantile(0.75)
        iqr = q3 - q1
        
        # Outliers: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        short_text = (lengths < lower_bound).sum()
        long_text = (lengths > upper_bound).sum()
        
        if short_text > 0 or long_text > 0:
            self.issues["outliers"].append(
                f"Text length outliers: {short_text} very short, {long_text} very long"
            )
            print(f"⚠️  Found {short_text} very short and {long_text} very long records")
        else:
            print("✅ Text lengths appear normal")

    def check_label_balance(self, df: pd.DataFrame, label_col: str) -> None:
        """Check class balance in sentiment labels."""
        print("🔍 Checking label distribution...")
        label_dist = df[label_col].value_counts().sort_index()
        
        if len(label_dist) != 2:
            self.issues["warnings"].append(
                f"Expected 2 classes (0, 1), found {len(label_dist)}: {label_dist.to_dict()}"
            )
        
        # Check for severe class imbalance (e.g., >80% one class)
        percentages = (label_dist / len(df) * 100).to_dict()
        for label, pct in percentages.items():
            if pct > 80:
                self.issues["warnings"].append(
                    f"Class imbalance: {label} represents {pct:.1f}% of data"
                )
        
        print(f"Label distribution: {label_dist.to_dict()}")

    def generate_summary(self, df: pd.DataFrame) -> None:
        """Generate summary statistics."""
        self.summary = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def write_report(self) -> None:
        """Write validation report to file."""
        lines = [
            "=" * 70,
            "DATA VALIDATION REPORT",
            "=" * 70,
            "",
            "DATASET SUMMARY",
            "-" * 70,
            f"Total Records: {self.summary.get('total_records', 'N/A')}",
            f"Total Columns: {self.summary.get('total_columns', 'N/A')}",
            f"Memory Usage: {self.summary.get('memory_usage_mb', 0):.2f} MB",
            f"Columns: {', '.join(self.summary.get('columns', []))}",
            "",
            "DATA TYPES",
            "-" * 70,
        ]
        
        for col, dtype in self.summary.get("dtypes", {}).items():
            lines.append(f"  {col}: {dtype}")
        
        lines.extend([
            "",
            "VALIDATION RESULTS",
            "-" * 70,
        ])
        
        # Schema errors
        if self.issues["schema_errors"]:
            lines.append(f"❌ SCHEMA ERRORS ({len(self.issues['schema_errors'])})")
            for err in self.issues["schema_errors"][:10]:  # First 10
                lines.append(f"   {err}")
            if len(self.issues["schema_errors"]) > 10:
                lines.append(f"   ... and {len(self.issues['schema_errors']) - 10} more")
            lines.append("")
        
        # Missing values
        if self.issues["missing_values"]:
            lines.append("⚠️  MISSING VALUES")
            for col, count in self.issues["missing_values"].items():
                pct = count / self.summary.get('total_records', 1) * 100
                lines.append(f"   {col}: {count} ({pct:.2f}%)")
            lines.append("")
        
        # Duplicates
        if self.issues["duplicates"] > 0:
            pct = self.issues["duplicates"] / self.summary.get('total_records', 1) * 100
            lines.append(f"⚠️  DUPLICATE RECORDS: {self.issues['duplicates']} ({pct:.2f}%)")
            lines.append("")
        
        # Encoding issues
        if self.issues["encoding_issues"]:
            lines.append("⚠️  ENCODING ISSUES")
            for issue in self.issues["encoding_issues"]:
                lines.append(f"   {issue}")
            lines.append("")
        
        # Outliers
        if self.issues["outliers"]:
            lines.append("⚠️  OUTLIERS & ANOMALIES")
            for issue in self.issues["outliers"]:
                lines.append(f"   {issue}")
            lines.append("")
        
        # Warnings
        if self.issues["warnings"]:
            lines.append("⚠️  WARNINGS")
            for warning in self.issues["warnings"]:
                lines.append(f"   {warning}")
            lines.append("")
        
        # Overall status
        lines.extend([
            "",
            "OVERALL STATUS",
            "-" * 70,
        ])
        
        has_critical_issues = bool(self.issues["schema_errors"])
        status = "❌ FAILED - Critical issues found" if has_critical_issues else "✅ PASSED"
        lines.append(status)
        
        if self.issues["missing_values"] or self.issues["duplicates"] > 0 or self.issues["outliers"]:
            lines.append("⚠️  Data quality issues detected - review recommended")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 70,
            "1. Address any schema violations before proceeding",
            "2. Review records with missing values",
            "3. Decide on duplicate handling strategy (remove, keep, merge)",
            "4. Examine outliers - may be legitimate edge cases or errors",
            "5. Consider class rebalancing if imbalance is extreme",
            "6. Re-run validation after any data cleaning",
            "",
            "=" * 70,
        ])
        
        report_text = "\n".join(lines)
        VALIDATION_REPORT.write_text(report_text, encoding="utf-8")
        print(f"\n📄 Validation report saved to: {VALIDATION_REPORT}")

    def write_issues_json(self) -> None:
        """Write detailed issues to JSON for programmatic access."""
        VALIDATION_ISSUES.write_text(
            json.dumps(self.issues, indent=2, default=str),
            encoding="utf-8"
        )
        print(f"📊 Issues JSON saved to: {VALIDATION_ISSUES}")


# ─── Main Validation Workflows ─────────────────────────────

def validate_csv_data(path: Path) -> tuple[pd.DataFrame, DataValidator]:
    """Complete validation pipeline for CSV data."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATING CSV DATA: {path.name}")
    print(f"{'=' * 70}\n")
    
    df = load_csv(path)
    validator = DataValidator()
    
    # Run all checks
    validator.validate_csv(df)
    validator.check_missing_values(df)
    validator.check_duplicates(df, subset=["review_id"])
    validator.check_encoding(df, "review_text")
    validator.check_text_length_outliers(df, "review_text")
    validator.check_label_balance(df, "sentiment")
    validator.generate_summary(df)
    
    # Generate reports
    validator.write_report()
    validator.write_issues_json()
    
    return df, validator


def validate_txt_data(path: Path) -> tuple[pd.DataFrame, DataValidator]:
    """Complete validation pipeline for text data."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATING TEXT DATA: {path.name}")
    print(f"{'=' * 70}\n")
    
    df = load_txt(path)
    validator = DataValidator()
    
    # Run all checks
    validator.validate_txt(df)
    validator.check_missing_values(df)
    validator.check_duplicates(df, subset=["review", "label"])
    validator.check_encoding(df, "review")
    validator.check_text_length_outliers(df, "review")
    validator.check_label_balance(df, "label")
    validator.generate_summary(df)
    
    # Generate reports
    validator.write_report()
    validator.write_issues_json()
    
    return df, validator


def main() -> None:
    """Run validation on available data files."""
    
    # Validate CSV if it exists
    if DATA_RAW_CSV.exists():
        df_csv, validator_csv = validate_csv_data(DATA_RAW_CSV)
        print("\n✅ CSV validation complete\n")
    else:
        print(f"⚠️  CSV file not found: {DATA_RAW_CSV}")
    
    # Validate text if it exists
    if DATA_RAW_TXT.exists():
        df_txt, validator_txt = validate_txt_data(DATA_RAW_TXT)
        print("\n✅ Text validation complete\n")
    else:
        print(f"⚠️  Text file not found: {DATA_RAW_TXT}")
    
    print("\n" + "=" * 70)
    print("🎉 Data validation pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
