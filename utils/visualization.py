"""
Automated Visualization Module for Voice2Data AI.
Generates responsive, dark-themed interactive Plotly charts dynamically
based on the shape, types, semantics, and intent of the executed query.
"""
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DARK_THEME_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="Inter, -apple-system, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.12)", tickfont=dict(color="#94A3B8")),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.12)", tickfont=dict(color="#94A3B8")),
    hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="Inter", bordercolor="rgba(255, 255, 255, 0.15)"),
    colorway=["#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"]
)


def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Identifies numeric, categorical, and datetime columns."""
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    
    date_cols = []
    cat_cols = []
    
    for c in df.columns:
        if c in num_cols:
            continue
        if "date" in c.lower() or "time" in c.lower():
            date_cols.append(c)
        else:
            cat_cols.append(c)
            
    return num_cols, cat_cols, date_cols


def _resolve_target_metric(df: pd.DataFrame, intent: Optional[Dict[str, Any]], title_hint: Optional[str]) -> Optional[str]:
    """
    Identifies the semantically correct numeric metric column based on intent and query hints.
    Guarantees that operations on 'Revenue' will visualize 'Revenue', not arbitrary columns like 'Units'.
    """
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    if not num_cols:
        return None

    if intent and isinstance(intent, dict):
        op = intent.get("operation", "")

        # 1. Top N / Bottom N
        if op in ("top_n", "bottom_n"):
            col = intent.get("column")
            if col in df.columns and col in num_cols:
                return col

        # 2. Groupby / Aggregate
        elif op in ("groupby", "aggregate"):
            aggs = intent.get("aggregations", [])
            for agg in aggs:
                col = agg.get("column") if isinstance(agg, dict) else None
                if col:
                    # Check exact or suffixed column in df
                    for c in num_cols:
                        if c == col or c.startswith(f"{col}_") or col.lower() in c.lower():
                            return c

        # 3. Sort
        elif op == "sort":
            sort_by = intent.get("sort_by", [])
            for s in sort_by:
                col = s.get("column") if isinstance(s, dict) else None
                if col in num_cols:
                    return col

        # 4. Filter Sort Top
        elif op == "filter_sort_top":
            sort_col = intent.get("sort_column")
            if sort_col in num_cols:
                return sort_col
            for f in intent.get("filters", []):
                col = f.get("column") if isinstance(f, dict) else None
                if col in num_cols:
                    return col

    # 5. Check if query/title_hint mentions a specific numeric column
    if title_hint:
        hint_lower = title_hint.lower()
        # Sort num_cols by length descending to match full names first
        for col in sorted(num_cols, key=lambda x: len(x), reverse=True):
            if col.lower() in hint_lower:
                return col

    # 6. Fallback: prioritize business metric keywords (revenue, sales, amount) over counters
    for col in num_cols:
        c_lower = col.lower()
        if any(k in c_lower for k in ("revenue", "sales", "amount", "total", "price", "profit")):
            return col

    return num_cols[0]


def _resolve_target_category(df: pd.DataFrame, intent: Optional[Dict[str, Any]], metric_col: Optional[str]) -> Optional[str]:
    """Identifies the best categorical/entity column to pair with the metric."""
    _, cat_cols, _ = _detect_columns(df)
    if not cat_cols:
        # If no explicit categorical column, check for non-metric columns
        non_metric = [c for c in df.columns if c != metric_col]
        return non_metric[0] if non_metric else None

    if intent and isinstance(intent, dict):
        # Check groupby
        group_by = intent.get("group_by")
        if isinstance(group_by, list) and len(group_by) > 0 and group_by[0] in df.columns:
            return group_by[0]
        elif isinstance(group_by, str) and group_by in df.columns:
            return group_by

    # Prioritize natural entities (Customer, Product, Region, Category, Transaction_ID)
    priority_order = ["transaction_id", "transaction", "customer", "product", "region", "category", "salesperson", "payment_mode"]
    for prio in priority_order:
        for col in cat_cols:
            if prio == col.lower() or prio in col.lower():
                return col

    return cat_cols[0]


def create_auto_chart(
    df: pd.DataFrame,
    title_hint: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None
) -> Optional[go.Figure]:
    """
    Analyzes DataFrame and intent to generate semantically accurate, dark-themed Plotly charts.
    """
    if df is None or df.empty or len(df.columns) < 1:
        return None

    num_cols, cat_cols, date_cols = _detect_columns(df)
    metric_col = _resolve_target_metric(df, intent, title_hint)
    cat_col = _resolve_target_category(df, intent, metric_col)

    op = intent.get("operation", "") if intent else ""

    # Case 1: Top N / Bottom N operations -> Direct Ranked Bar Chart of target metric
    if op in ("top_n", "bottom_n") and metric_col and cat_col:
        is_asc = bool(intent.get("ascending", op == "bottom_n"))
        df_plot = df.sort_values(by=metric_col, ascending=is_asc)
        
        fig = px.bar(
            df_plot,
            x=cat_col,
            y=metric_col,
            title=title_hint or f"{'Bottom' if is_asc else 'Top'} {len(df_plot)} by {metric_col}",
            color=metric_col,
            color_continuous_scale="Blues" if is_asc else "Viridis",
            text=metric_col
        )
        fig.update_traces(texttemplate='%{text:,.2s}', textposition='outside')
        fig.update_layout(**DARK_THEME_LAYOUT)
        fig.update_coloraxes(showscale=False)
        return fig

    # Case 2: Groupby & Aggregations
    if op == "groupby" and metric_col and cat_col:
        if df[cat_col].nunique() <= 6 and (df[metric_col] >= 0).all():
            fig = px.pie(
                df,
                names=cat_col,
                values=metric_col,
                hole=0.55,
                title=title_hint or f"{metric_col} Distribution by {cat_col}",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
            fig.update_layout(**DARK_THEME_LAYOUT)
            return fig
        else:
            df_sorted = df.sort_values(by=metric_col, ascending=False)
            fig = px.bar(
                df_sorted,
                x=cat_col,
                y=metric_col,
                title=title_hint or f"{metric_col} by {cat_col}",
                color=cat_col,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
            )
            fig.update_layout(**DARK_THEME_LAYOUT, showlegend=False)
            return fig

    # Case 3: Time Series (Date + Selected Numeric Metric)
    if date_cols and metric_col and (op == "date_filter" or "date" in str(title_hint).lower() or len(df) > 20):
        date_col = date_cols[0]
        try:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
            df_plot = df_plot.dropna(subset=[date_col]).sort_values(by=date_col)
            
            if df_plot[date_col].duplicated().any():
                df_plot = df_plot.groupby(date_col)[metric_col].sum().reset_index()

            fig = px.line(
                df_plot,
                x=date_col,
                y=metric_col,
                title=title_hint or f"Trend: {metric_col} over Time",
                markers=True,
                line_shape="spline"
            )
            fig.update_traces(line=dict(width=3, color="#38bdf8"), marker=dict(size=6, color="#818cf8"))
            fig.update_layout(**DARK_THEME_LAYOUT)
            return fig
        except Exception:
            pass

    # Case 4: Categorical + Numeric Metric
    if cat_col and metric_col:
        if len(df) > 8:
            df_sorted = df.sort_values(by=metric_col, ascending=True).tail(20)
            fig = px.bar(
                df_sorted,
                x=metric_col,
                y=cat_col,
                orientation="h",
                title=title_hint or f"{cat_col} by {metric_col}",
                color=metric_col,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(**DARK_THEME_LAYOUT)
            fig.update_coloraxes(showscale=False)
            return fig
        else:
            fig = px.bar(
                df,
                x=cat_col,
                y=metric_col,
                title=title_hint or f"{metric_col} by {cat_col}",
                color=cat_col,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
            )
            fig.update_layout(**DARK_THEME_LAYOUT, showlegend=False)
            return fig

    # Case 5: Single Numeric Column (Histogram)
    if metric_col:
        fig = px.histogram(
            df,
            x=metric_col,
            nbins=30,
            title=title_hint or f"Distribution of {metric_col}",
            color_discrete_sequence=["#38bdf8"]
        )
        fig.update_layout(**DARK_THEME_LAYOUT)
        return fig

    # Case 6: Categorical counts
    if cat_cols:
        cat_c = cat_cols[0]
        counts_df = df[cat_c].value_counts().reset_index()
        counts_df.columns = [cat_c, "Count"]
        fig = px.bar(
            counts_df.head(15),
            x=cat_c,
            y="Count",
            title=title_hint or f"Count by {cat_c}",
            color="Count",
            color_continuous_scale="Teal"
        )
        fig.update_layout(**DARK_THEME_LAYOUT)
        fig.update_coloraxes(showscale=False)
        return fig

    return None
