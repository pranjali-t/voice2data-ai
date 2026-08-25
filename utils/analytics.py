"""
Analytics and KPI helper module for Voice2Data AI.
Calculates dashboard metrics, deltas, and dataset summaries.
"""
from typing import Dict, Any, Optional
import pandas as pd


def compute_kpis(
    current_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
    previous_df: Optional[pd.DataFrame] = None,
    execution_count: int = 0
) -> Dict[str, Any]:
    """
    Computes real-time KPI metrics with deltas for the Streamlit dashboard.
    """
    if current_df is None or current_df.empty:
        return {
            "rows": 0, "rows_delta": 0,
            "cols": 0, "cols_delta": 0,
            "missing": 0, "missing_delta": 0,
            "duplicates": 0,
            "operations": execution_count,
            "operations_delta": 1 if execution_count > 0 else 0
        }

    curr_rows = len(current_df)
    curr_cols = len(current_df.columns)
    curr_missing = int(current_df.isna().sum().sum())
    curr_duplicates = int(current_df.duplicated().sum())

    # Baseline for deltas
    baseline_df = previous_df if previous_df is not None else original_df

    if baseline_df is not None:
        base_rows = len(baseline_df)
        base_cols = len(baseline_df.columns)
        base_missing = int(baseline_df.isna().sum().sum())

        rows_delta = curr_rows - base_rows
        cols_delta = curr_cols - base_cols
        missing_delta = curr_missing - base_missing
    else:
        rows_delta = 0
        cols_delta = 0
        missing_delta = 0

    return {
        "rows": curr_rows,
        "rows_delta": f"{rows_delta:+d}" if rows_delta != 0 else None,
        "cols": curr_cols,
        "cols_delta": f"{cols_delta:+d}" if cols_delta != 0 else None,
        "missing": curr_missing,
        "missing_delta": f"{missing_delta:+d}" if missing_delta != 0 else None,
        "duplicates": curr_duplicates,
        "operations": execution_count,
        "operations_delta": "+1" if execution_count > 0 else None
    }


def get_quick_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Extracts high-level business insights from active DataFrame."""
    insights = {}
    if df is None or df.empty:
        return insights

    # Check for revenue/sales column
    for col in df.columns:
        if "revenue" in col.lower() or "sales" in col.lower() or "amount" in col.lower():
            if pd.api.types.is_numeric_dtype(df[col]):
                insights["Total Value"] = f"${df[col].sum():,.2f}"
                insights["Avg Value"] = f"${df[col].mean():,.2f}"
                break

    # Check for date range
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                dt_series = pd.to_datetime(df[col], errors='coerce').dropna()
                if not dt_series.empty:
                    insights["Date Span"] = f"{dt_series.min().strftime('%b %Y')} → {dt_series.max().strftime('%b %Y')}"
                    break
            except Exception:
                pass

    return insights
