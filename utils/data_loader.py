"""
Data loading and dataset quality profiling utility for Voice2Data AI.
Supports CSV, Excel (XLSX/XLS) loading with robust error handling and schema profiling.
"""
import os
from typing import Tuple, Optional, Dict, Any, Union
import io
import pandas as pd


DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_sales.csv")


def load_default_dataset() -> pd.DataFrame:
    """Loads the pre-packaged sample_sales.csv dataset safely."""
    # Check data/sample_sales.csv first
    if os.path.exists(DEFAULT_DATASET_PATH):
        return pd.read_csv(DEFAULT_DATASET_PATH)
    
    # Fallback to root sample_sales.csv if needed
    root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_sales.csv")
    if os.path.exists(root_path):
        return pd.read_csv(root_path)

    raise FileNotFoundError("Default dataset 'sample_sales.csv' not found.")


def load_uploaded_file(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely reads an uploaded CSV or Excel file into a pandas DataFrame.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file-like buffer.
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    filename = getattr(uploaded_file, "name", "dataset.csv").lower()

    try:
        if filename.endswith(".csv"):
            # Try utf-8 first, fallback to latin-1
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif filename.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            return None, f"Unsupported file extension for '{filename}'. Please upload a .csv or .xlsx file."

        if df.empty:
            return None, "Uploaded dataset is empty."

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        return df, None

    except Exception as e:
        return None, f"Failed to parse uploaded dataset: {str(e)}"


def get_dataset_quality_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured data quality profile including KPIs and a column-level schema table.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "missing_values": 0,
            "missing_pct": 0.0,
            "duplicate_rows": 0,
            "memory_usage_kb": 0.0,
            "schema_df": pd.DataFrame()
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_values = int(df.isna().sum().sum())
    missing_pct = round((missing_values / (total_rows * total_cols)) * 100, 2) if total_rows * total_cols > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

    schema_records = []
    for col in df.columns:
        col_series = df[col]
        null_cnt = int(col_series.isna().sum())
        null_ratio = f"{round((null_cnt / total_rows) * 100, 1)}%" if total_rows > 0 else "0%"
        unique_cnt = int(col_series.nunique(dropna=True))
        sample_val = str(col_series.dropna().iloc[0]) if not col_series.dropna().empty else "-"
        if len(sample_val) > 30:
            sample_val = sample_val[:27] + "..."

        schema_records.append({
            "Column": col,
            "Data Type": str(col_series.dtype),
            "Missing": null_cnt,
            "Missing %": null_ratio,
            "Unique Values": unique_cnt,
            "Sample Value": sample_val
        })

    schema_df = pd.DataFrame(schema_records)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_values": missing_values,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "memory_usage_kb": memory_kb,
        "schema_df": schema_df
    }
