"""
Controlled Pandas Execution Engine for Voice2Data AI.
Executes strictly validated structured intents using explicit, safe Pandas APIs.
NEVER uses eval(), exec(), or arbitrary code execution.
"""
from typing import Dict, Any, Tuple, Optional, List
import time
import pandas as pd
import numpy as np


class PandasEngine:
    """Safe, deterministic execution engine for pandas operations."""

    @staticmethod
    def _coerce_type_for_comparison(series: pd.Series, value: Any) -> Tuple[pd.Series, Any]:
        """Safely casts types for comparison without raising unhandled errors."""
        if pd.api.types.is_numeric_dtype(series):
            try:
                if isinstance(value, str):
                    clean_val = value.replace(",", "").replace("$", "").replace("₹", "").strip()
                    val_num = float(clean_val)
                    if series.dtype == 'int64' and val_num.is_integer():
                        val_num = int(val_num)
                    return series, val_num
                return series, float(value)
            except Exception:
                return series.astype(str), str(value)
        elif pd.api.types.is_datetime64_any_dtype(series):
            try:
                val_dt = pd.to_datetime(value)
                return series, val_dt
            except Exception:
                return series.astype(str), str(value)
        else:
            return series.astype(str), str(value)

    @classmethod
    def apply_filter(cls, df: pd.DataFrame, conditions: List[Dict[str, Any]], logical_op: str = "AND") -> pd.DataFrame:
        """Applies validated filter conditions using boolean masks."""
        if not conditions:
            return df.copy()

        masks = []
        for cond in conditions:
            col = cond["column"]
            op = cond["operator"]
            val = cond["value"]
            series = df[col]

            if op == "==":
                if isinstance(val, str) and not pd.api.types.is_numeric_dtype(series):
                    mask = series.astype(str).str.strip().str.lower() == str(val).strip().lower()
                else:
                    s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                    mask = s_coerced == v_coerced
            elif op == "!=":
                if isinstance(val, str) and not pd.api.types.is_numeric_dtype(series):
                    mask = series.astype(str).str.strip().str.lower() != str(val).strip().lower()
                else:
                    s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                    mask = s_coerced != v_coerced
            elif op == ">":
                s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                mask = s_coerced > v_coerced
            elif op == "<":
                s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                mask = s_coerced < v_coerced
            elif op == ">=":
                s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                mask = s_coerced >= v_coerced
            elif op == "<=":
                s_coerced, v_coerced = cls._coerce_type_for_comparison(series, val)
                mask = s_coerced <= v_coerced
            elif op == "contains":
                mask = series.astype(str).str.contains(str(val), case=False, na=False)
            elif op == "in":
                if isinstance(val, list):
                    val_set = {str(x).strip().lower() for x in val}
                    mask = series.astype(str).str.strip().str.lower().isin(val_set)
                else:
                    mask = series.astype(str).str.strip().str.lower() == str(val).strip().lower()
            elif op == "between":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    s_coerced, v0 = cls._coerce_type_for_comparison(series, val[0])
                    _, v1 = cls._coerce_type_for_comparison(series, val[1])
                    mask = (s_coerced >= v0) & (s_coerced <= v1)
                else:
                    mask = pd.Series([True] * len(df), index=df.index)
            else:
                mask = pd.Series([True] * len(df), index=df.index)

            masks.append(mask)

        if not masks:
            return df.copy()

        if logical_op.upper() == "OR":
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask | m
        else:
            final_mask = masks[0]
            for m in masks[1:]:
                final_mask = final_mask & m

        return df[final_mask].copy()

    @classmethod
    def apply_sort(cls, df: pd.DataFrame, sort_by: List[Dict[str, Any]]) -> pd.DataFrame:
        """Sorts the dataframe by one or more columns."""
        cols = [item["column"] for item in sort_by]
        ascending = [item.get("ascending", True) for item in sort_by]
        return df.sort_values(by=cols, ascending=ascending).copy()

    @classmethod
    def apply_top_n(cls, df: pd.DataFrame, column: str, n: int, ascending: bool = False,
                    group_by: Optional[str] = None, agg_func: str = "sum") -> pd.DataFrame:
        """Retrieves top N or bottom N rows, optionally aggregating by an entity column."""
        if group_by:
            if not pd.api.types.is_numeric_dtype(df[column]):
                # Attempt to convert or fallback to count
                try:
                    df_temp = df.copy()
                    df_temp[column] = pd.to_numeric(df_temp[column], errors='coerce')
                    grouped = df_temp.groupby(group_by)[column].agg(agg_func).reset_index()
                except Exception:
                    grouped = df.groupby(group_by)[column].count().reset_index()
            else:
                grouped = df.groupby(group_by)[column].agg(agg_func).reset_index()

            res = grouped.sort_values(by=column, ascending=ascending).head(n)
            return res.copy()
        else:
            return df.sort_values(by=column, ascending=ascending).head(n).copy()

    @classmethod
    def apply_groupby(cls, df: pd.DataFrame, group_by: List[str], aggregations: List[Dict[str, str]],
                      sort_column: Optional[str] = None, ascending: bool = False,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Groups dataset by categorical columns and calculates specified aggregations."""
        agg_dict = {}
        for item in aggregations:
            col = item["column"]
            func = item["agg_func"]
            if func == "mean":
                func = "mean"
            if col not in agg_dict:
                agg_dict[col] = []
            if func not in agg_dict[col]:
                agg_dict[col].append(func)

        # Simplify agg_dict if each column has single function
        single_agg_dict = {col: funcs[0] if len(funcs) == 1 else funcs for col, funcs in agg_dict.items()}

        grouped = df.groupby(group_by).agg(single_agg_dict)

        # Flatten multi-index columns if present
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"{col}_{func}" if func else col for col, func in grouped.columns]
        grouped = grouped.reset_index()

        if sort_column:
            sort_targets = [c for c in grouped.columns if sort_column.lower() in c.lower()]
            if sort_targets:
                grouped = grouped.sort_values(by=sort_targets[0], ascending=ascending)
            elif len(grouped.columns) > len(group_by):
                grouped = grouped.sort_values(by=grouped.columns[-1], ascending=ascending)

        if limit and limit > 0:
            grouped = grouped.head(limit)

        return grouped.copy()

    @classmethod
    def apply_aggregate(cls, df: pd.DataFrame, aggregations: List[Dict[str, str]]) -> pd.DataFrame:
        """Aggregates metrics across the entire dataset returning a single-row summary."""
        result_dict = {}
        for item in aggregations:
            col = item["column"]
            func = item["agg_func"]
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if func == "sum":
                    val = series.sum()
                elif func in ("mean", "avg"):
                    val = series.mean()
                elif func == "count":
                    val = series.count()
                elif func == "min":
                    val = series.min()
                elif func == "max":
                    val = series.max()
                elif func == "median":
                    val = series.median()
                elif func == "std":
                    val = series.std()
                elif func == "nunique":
                    val = series.nunique()
                else:
                    val = series.count()
            else:
                if func == "nunique":
                    val = series.nunique()
                else:
                    val = series.count()
            
            label = f"{col} ({func.capitalize()})"
            result_dict[label] = [round(val, 2) if isinstance(val, (int, float, np.floating, np.integer)) else val]

        return pd.DataFrame(result_dict)

    @classmethod
    def apply_select_columns(cls, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Projects only the requested subset of columns."""
        valid_cols = [c for c in columns if c in df.columns]
        return df[valid_cols].copy()

    @classmethod
    def apply_rename_columns(cls, df: pd.DataFrame, columns_map: Dict[str, str]) -> pd.DataFrame:
        """Safely renames columns."""
        return df.rename(columns=columns_map).copy()

    @classmethod
    def apply_date_filter(cls, df: pd.DataFrame, date_column: str,
                          start_date: Optional[str] = None, end_date: Optional[str] = None,
                          year: Optional[int] = None, quarter: Optional[str] = None,
                          month: Optional[int] = None) -> pd.DataFrame:
        """Filters data based on datetime ranges or date attributes."""
        dt_series = pd.to_datetime(df[date_column], errors='coerce')
        mask = pd.Series([True] * len(df), index=df.index)

        if start_date:
            try:
                s_dt = pd.to_datetime(start_date)
                mask = mask & (dt_series >= s_dt)
            except Exception:
                pass

        if end_date:
            try:
                e_dt = pd.to_datetime(end_date)
                mask = mask & (dt_series <= e_dt)
            except Exception:
                pass

        if year:
            try:
                mask = mask & (dt_series.dt.year == int(year))
            except Exception:
                pass

        if quarter:
            q_clean = str(quarter).upper().replace("Q", "").strip()
            try:
                q_int = int(q_clean)
                mask = mask & (dt_series.dt.quarter == q_int)
            except Exception:
                pass

        if month:
            try:
                mask = mask & (dt_series.dt.month == int(month))
            except Exception:
                pass

        return df[mask].copy()

    @classmethod
    def apply_filter_sort_top(cls, df: pd.DataFrame, filters: List[Dict[str, Any]],
                              logical_op: str = "AND", sort_column: Optional[str] = None,
                              ascending: bool = False, limit: Optional[int] = None) -> pd.DataFrame:
        """Executes a composite operation: Filter -> Sort -> Slice limit."""
        res = cls.apply_filter(df, filters, logical_op)
        if sort_column and sort_column in res.columns:
            res = res.sort_values(by=sort_column, ascending=ascending)
        if limit and limit > 0:
            res = res.head(limit)
        return res.copy()


def execute_intent(df: pd.DataFrame, intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Master runner: safely executes a validated intent on a DataFrame.
    
    Returns a dictionary containing:
        - success: bool
        - result_df: pd.DataFrame
        - rows_before: int
        - rows_after: int
        - rows_affected: int
        - execution_time_ms: float
        - operation_summary: str
        - error: Optional[str]
    """
    start_time = time.perf_counter()
    rows_before = len(df)
    operation = intent.get("operation", "")

    try:
        if operation in ("filter", "value_filter"):
            conditions = intent.get("conditions", [])
            logical_op = intent.get("logical_operator", "AND")
            result_df = PandasEngine.apply_filter(df, conditions, logical_op)
            cond_desc = f" {logical_op} ".join([f"{c['column']} {c['operator']} {c['value']}" for c in conditions])
            summary = f"Filtered rows where {cond_desc}."

        elif operation == "sort":
            sort_by = intent.get("sort_by", [])
            result_df = PandasEngine.apply_sort(df, sort_by)
            sort_desc = ", ".join([f"{s['column']} ({'asc' if s.get('ascending', True) else 'desc'})" for s in sort_by])
            summary = f"Sorted dataset by {sort_desc}."

        elif operation in ("top_n", "bottom_n"):
            col = intent["column"]
            n = intent["n"]
            asc = intent["ascending"]
            group_by = intent.get("group_by")
            agg_func = intent.get("agg_func", "sum")
            result_df = PandasEngine.apply_top_n(df, column=col, n=n, ascending=asc, group_by=group_by, agg_func=agg_func)
            if group_by:
                summary = f"Grouped by {group_by} and retrieved {'bottom' if asc else 'top'} {n} records by {col} ({agg_func})."
            else:
                summary = f"Selected {'bottom' if asc else 'top'} {n} rows by {col}."

        elif operation == "groupby":
            group_by = intent["group_by"]
            aggregations = intent["aggregations"]
            sort_col = intent.get("sort_column")
            ascending = intent.get("ascending", False)
            limit = intent.get("limit")
            result_df = PandasEngine.apply_groupby(df, group_by, aggregations, sort_col, ascending, limit)
            agg_desc = ", ".join([f"{a['agg_func']}({a['column']})" for a in aggregations])
            summary = f"Grouped by {', '.join(group_by)} with aggregations: {agg_desc}."

        elif operation == "aggregate":
            aggregations = intent["aggregations"]
            result_df = PandasEngine.apply_aggregate(df, aggregations)
            agg_desc = ", ".join([f"{a['agg_func']}({a['column']})" for a in aggregations])
            summary = f"Computed dataset metrics: {agg_desc}."

        elif operation == "select_columns":
            cols = intent["columns"]
            result_df = PandasEngine.apply_select_columns(df, cols)
            summary = f"Selected columns: {', '.join(cols)}."

        elif operation == "rename_columns":
            col_map = intent["columns_map"]
            result_df = PandasEngine.apply_rename_columns(df, col_map)
            rename_desc = ", ".join([f"'{k}' → '{v}'" for k, v in col_map.items()])
            summary = f"Renamed columns: {rename_desc}."

        elif operation == "date_filter":
            date_col = intent["date_column"]
            start_date = intent.get("start_date")
            end_date = intent.get("end_date")
            year = intent.get("year")
            quarter = intent.get("quarter")
            month = intent.get("month")
            result_df = PandasEngine.apply_date_filter(df, date_col, start_date, end_date, year, quarter, month)
            summary = f"Filtered date on column '{date_col}'."

        elif operation == "filter_sort_top":
            filters = intent.get("filters", [])
            logical_op = intent.get("logical_operator", "AND")
            sort_col = intent.get("sort_column")
            ascending = intent.get("ascending", False)
            limit = intent.get("limit")
            result_df = PandasEngine.apply_filter_sort_top(df, filters, logical_op, sort_col, ascending, limit)
            summary = f"Executed composite filter, sort by {sort_col or 'default'}, limit {limit or 'all'}."

        else:
            return {
                "success": False,
                "result_df": df.copy(),
                "rows_before": rows_before,
                "rows_after": rows_before,
                "rows_affected": 0,
                "execution_time_ms": 0.0,
                "operation_summary": f"Unsupported operation '{operation}'.",
                "error": f"Operation '{operation}' is not supported by the Pandas engine."
            }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        rows_after = len(result_df)
        rows_affected = abs(rows_before - rows_after)

        return {
            "success": True,
            "result_df": result_df,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_affected": rows_affected,
            "execution_time_ms": elapsed_ms,
            "operation_summary": summary,
            "error": None
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "success": False,
            "result_df": df.copy(),
            "rows_before": rows_before,
            "rows_after": rows_before,
            "rows_affected": 0,
            "execution_time_ms": elapsed_ms,
            "operation_summary": f"Execution failed: {str(e)}",
            "error": str(e)
        }
