"""
Validator module for Voice2Data AI.
Performs deterministic schema, type, column, and security checks on Gemini intents
prior to passing them to the Pandas execution engine.
"""
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd


ALLOWED_OPERATIONS = {
    "filter",
    "value_filter",
    "sort",
    "top_n",
    "bottom_n",
    "groupby",
    "aggregate",
    "select_columns",
    "rename_columns",
    "date_filter",
    "filter_sort_top",
    "unsupported"
}

ALLOWED_OPERATORS = {"==", "!=", ">", "<", ">=", "<=", "contains", "in", "between"}
ALLOWED_AGG_FUNCS = {"sum", "mean", "avg", "count", "min", "max", "median", "std", "nunique"}
FORBIDDEN_KEYWORDS = {"eval", "exec", "import", "__", "lambda", "globals", "locals", "os.", "sys.", "subprocess"}


def _check_forbidden_payload(payload: Any) -> bool:
    """Recursively checks if any string in the intent contains dangerous keywords."""
    if isinstance(payload, str):
        lowered = payload.lower()
        for kw in FORBIDDEN_KEYWORDS:
            if kw in lowered:
                return True
    elif isinstance(payload, dict):
        for k, v in payload.items():
            if _check_forbidden_payload(k) or _check_forbidden_payload(v):
                return True
    elif isinstance(payload, list):
        for item in payload:
            if _check_forbidden_payload(item):
                return True
    return False


def validate_intent(intent: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validates a structured intent dictionary against the actual pandas DataFrame.
    
    Args:
        intent: Structured JSON from Gemini.
        df: The active pandas DataFrame.
        
    Returns:
        Tuple of (is_valid, validated_intent, error_message)
    """
    if not isinstance(intent, dict):
        return False, {}, "Intent must be a JSON object (dictionary)."

    if _check_forbidden_payload(intent):
        return False, {}, "Security Violation: Forbidden code or keyword detected in intent."

    operation = intent.get("operation")
    if not operation or not isinstance(operation, str):
        return False, {}, "Intent missing required 'operation' field."

    operation = operation.strip().lower()
    intent["operation"] = operation

    if operation not in ALLOWED_OPERATIONS:
        return False, {}, f"Unsupported operation '{operation}'. Supported operations: {', '.join(sorted(ALLOWED_OPERATIONS))}."

    # If the LLM already classified as unsupported
    if operation == "unsupported":
        reason = intent.get("reason", "Operation is not supported for this dataset.")
        return False, intent, str(reason)

    if df is None or df.empty:
        return False, {}, "Cannot execute operation on an empty or missing dataset."

    df_columns = list(df.columns)
    df_columns_lower = {col.lower(): col for col in df_columns}

    def _resolve_col(col_name: Any) -> Optional[str]:
        if not isinstance(col_name, str):
            return None
        if col_name in df_columns:
            return col_name
        if col_name.lower() in df_columns_lower:
            return df_columns_lower[col_name.lower()]
        return None

    # 1. FILTER / VALUE_FILTER
    if operation in ("filter", "value_filter"):
        conditions = intent.get("conditions", [])
        if not isinstance(conditions, list) or len(conditions) == 0:
            return False, {}, "Filter operation requires a non-empty list of 'conditions'."
        
        logical_op = str(intent.get("logical_operator", "AND")).upper()
        if logical_op not in ("AND", "OR"):
            logical_op = "AND"
        intent["logical_operator"] = logical_op

        validated_conditions = []
        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                return False, {}, f"Condition #{i+1} must be an object."
            
            raw_col = cond.get("column")
            col = _resolve_col(raw_col)
            if not col:
                return False, {}, f"Column '{raw_col}' does not exist in dataset. Available columns: {', '.join(df_columns)}"
            
            op = str(cond.get("operator", "==")).strip()
            if op not in ALLOWED_OPERATORS:
                return False, {}, f"Invalid operator '{op}' in condition #{i+1}. Allowed operators: {', '.join(ALLOWED_OPERATORS)}"
            
            val = cond.get("value")
            if val is None and op not in ("==", "!="):
                return False, {}, f"Missing value for condition on column '{col}'."

            validated_conditions.append({
                "column": col,
                "operator": op,
                "value": val
            })
        intent["conditions"] = validated_conditions
        return True, intent, None

    # 2. SORT
    elif operation == "sort":
        sort_by = intent.get("sort_by", [])
        if isinstance(sort_by, dict):
            sort_by = [sort_by]
        if not isinstance(sort_by, list) or len(sort_by) == 0:
            # Fallback check if top-level column is passed
            raw_col = intent.get("column")
            if raw_col:
                sort_by = [{"column": raw_col, "ascending": intent.get("ascending", True)}]
            else:
                return False, {}, "Sort operation requires 'sort_by' list or 'column' field."
        
        validated_sort = []
        for i, item in enumerate(sort_by):
            if not isinstance(item, dict):
                return False, {}, f"Sort rule #{i+1} must be an object."
            raw_col = item.get("column")
            col = _resolve_col(raw_col)
            if not col:
                return False, {}, f"Sort column '{raw_col}' does not exist. Available columns: {', '.join(df_columns)}"
            ascending = bool(item.get("ascending", True))
            validated_sort.append({"column": col, "ascending": ascending})
        
        intent["sort_by"] = validated_sort
        return True, intent, None

    # 3. TOP_N / BOTTOM_N
    elif operation in ("top_n", "bottom_n"):
        raw_col = intent.get("column")
        col = _resolve_col(raw_col)
        if not col:
            return False, {}, f"Target column '{raw_col}' does not exist. Available columns: {', '.join(df_columns)}"
        
        try:
            n = int(intent.get("n", intent.get("limit", 10)))
            if n <= 0:
                return False, {}, f"Limit 'n' must be a positive integer, got {n}."
            if n > 100000:
                n = 100000
        except (ValueError, TypeError):
            return False, {}, "Invalid value for 'n'. Must be an integer."

        intent["column"] = col
        intent["n"] = n
        intent["ascending"] = True if operation == "bottom_n" else bool(intent.get("ascending", False))

        raw_group = intent.get("group_by")
        if raw_group:
            group_col = _resolve_col(raw_group)
            if not group_col:
                return False, {}, f"Group by column '{raw_group}' does not exist."
            intent["group_by"] = group_col
            agg_func = str(intent.get("agg_func", "sum")).lower()
            if agg_func == "avg":
                agg_func = "mean"
            if agg_func not in ALLOWED_AGG_FUNCS:
                agg_func = "sum"
            intent["agg_func"] = agg_func
        else:
            intent["group_by"] = None

        return True, intent, None

    # 4. GROUPBY
    elif operation == "groupby":
        group_by = intent.get("group_by", [])
        if isinstance(group_by, str):
            group_by = [group_by]
        if not isinstance(group_by, list) or len(group_by) == 0:
            return False, {}, "Groupby operation requires at least one column in 'group_by'."
        
        validated_group = []
        for col_name in group_by:
            c = _resolve_col(col_name)
            if not c:
                return False, {}, f"Groupby column '{col_name}' does not exist in dataset."
            validated_group.append(c)
        intent["group_by"] = validated_group

        aggregations = intent.get("aggregations", [])
        if isinstance(aggregations, dict):
            aggregations = [{"column": k, "agg_func": v} for k, v in aggregations.items()]
        if not isinstance(aggregations, list) or len(aggregations) == 0:
            # Fallback to sum of first numeric column if none provided
            num_cols = list(df.select_dtypes(include=['number']).columns)
            if num_cols:
                aggregations = [{"column": num_cols[0], "agg_func": "sum"}]
            else:
                aggregations = [{"column": df_columns[0], "agg_func": "count"}]

        validated_aggs = []
        for agg in aggregations:
            if not isinstance(agg, dict):
                continue
            raw_c = agg.get("column")
            c = _resolve_col(raw_c)
            if not c:
                return False, {}, f"Aggregation column '{raw_c}' does not exist in dataset."
            func = str(agg.get("agg_func", "sum")).lower()
            if func == "avg":
                func = "mean"
            if func not in ALLOWED_AGG_FUNCS:
                return False, {}, f"Aggregation function '{func}' is not supported. Allowed: {', '.join(ALLOWED_AGG_FUNCS)}"
            validated_aggs.append({"column": c, "agg_func": func})

        if not validated_aggs:
            return False, {}, "No valid aggregations specified."

        intent["aggregations"] = validated_aggs

        sort_col = intent.get("sort_column")
        if sort_col:
            res_sc = _resolve_col(sort_col)
            intent["sort_column"] = res_sc if res_sc else validated_aggs[0]["column"]
        intent["ascending"] = bool(intent.get("ascending", False))

        if intent.get("limit") is not None:
            try:
                intent["limit"] = max(1, int(intent["limit"]))
            except (ValueError, TypeError):
                intent["limit"] = None

        return True, intent, None

    # 5. AGGREGATE
    elif operation == "aggregate":
        aggregations = intent.get("aggregations", [])
        if isinstance(aggregations, dict):
            aggregations = [{"column": k, "agg_func": v} for k, v in aggregations.items()]
        if not isinstance(aggregations, list) or len(aggregations) == 0:
            return False, {}, "Aggregate operation requires 'aggregations' list."

        validated_aggs = []
        for agg in aggregations:
            if not isinstance(agg, dict):
                continue
            raw_c = agg.get("column")
            c = _resolve_col(raw_c)
            if not c:
                return False, {}, f"Aggregation column '{raw_c}' does not exist."
            func = str(agg.get("agg_func", "sum")).lower()
            if func == "avg":
                func = "mean"
            if func not in ALLOWED_AGG_FUNCS:
                return False, {}, f"Unsupported aggregation function '{func}'."
            validated_aggs.append({"column": c, "agg_func": func})

        intent["aggregations"] = validated_aggs
        return True, intent, None

    # 6. SELECT_COLUMNS
    elif operation == "select_columns":
        columns = intent.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list) or len(columns) == 0:
            return False, {}, "Select columns operation requires a list of 'columns'."

        validated_cols = []
        for col_name in columns:
            c = _resolve_col(col_name)
            if not c:
                return False, {}, f"Column '{col_name}' does not exist in dataset. Available: {', '.join(df_columns)}"
            validated_cols.append(c)

        intent["columns"] = validated_cols
        return True, intent, None

    # 7. RENAME_COLUMNS
    elif operation == "rename_columns":
        columns_map = intent.get("columns_map", {})
        if not isinstance(columns_map, dict) or len(columns_map) == 0:
            return False, {}, "Rename columns operation requires a 'columns_map' dictionary."

        validated_map = {}
        for old_col, new_col in columns_map.items():
            c = _resolve_col(old_col)
            if not c:
                return False, {}, f"Column '{old_col}' to rename does not exist."
            if not isinstance(new_col, str) or not new_col.strip():
                return False, {}, f"New column name for '{old_col}' must be a non-empty string."
            validated_map[c] = str(new_col).strip()

        intent["columns_map"] = validated_map
        return True, intent, None

    # 8. DATE_FILTER
    elif operation == "date_filter":
        raw_col = intent.get("date_column")
        date_col = _resolve_col(raw_col) if raw_col else None
        if not date_col:
            # Attempt to find first datetime or date-named column
            for c in df_columns:
                if "date" in c.lower() or "time" in c.lower():
                    date_col = c
                    break
        if not date_col:
            return False, {}, "No valid date column found in dataset for date filter."

        intent["date_column"] = date_col
        return True, intent, None

    # 9. FILTER_SORT_TOP (Composite)
    elif operation == "filter_sort_top":
        filters = intent.get("filters", [])
        if not isinstance(filters, list):
            filters = []
        
        validated_filters = []
        for f in filters:
            if not isinstance(f, dict):
                continue
            raw_c = f.get("column")
            c = _resolve_col(raw_c)
            if not c:
                return False, {}, f"Filter column '{raw_c}' does not exist in dataset."
            op = str(f.get("operator", "==")).strip()
            if op not in ALLOWED_OPERATORS:
                return False, {}, f"Invalid operator '{op}'."
            validated_filters.append({
                "column": c,
                "operator": op,
                "value": f.get("value")
            })
        intent["filters"] = validated_filters

        sort_col = intent.get("sort_column")
        if sort_col:
            sc = _resolve_col(sort_col)
            if not sc:
                return False, {}, f"Sort column '{sort_col}' does not exist in dataset."
            intent["sort_column"] = sc

        if intent.get("limit") is not None:
            try:
                intent["limit"] = max(1, int(intent["limit"]))
            except (ValueError, TypeError):
                intent["limit"] = None

        return True, intent, None

    return False, {}, f"Unrecognized operation '{operation}'."
