"""
Voice2Data AI - Natural Language to Safe Pandas Data Automation Engine.
A production-grade Streamlit application for safe, deterministic dataset automation.
"""
import os
import io
import time
from datetime import datetime
import pandas as pd
import streamlit as st

# Configure Streamlit page layout & branding
st.set_page_config(
    page_title="Voice2Data AI | Natural Language Data Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import engine & utility layers
from engine.gemini_engine import GeminiEngine, get_api_key
from engine.validator import validate_intent
from engine.pandas_engine import execute_intent
from utils.data_loader import load_default_dataset, load_uploaded_file, get_dataset_quality_profile
from utils.analytics import compute_kpis, get_quick_insights
from utils.visualization import create_auto_chart, DARK_THEME_LAYOUT
import plotly.express as px


# ---------------------------------------------------------
# Custom Modern SaaS Dark Theme CSS
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.98) 0%, rgba(10, 15, 30, 1) 90.2%);
        color: #f1f5f9;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Custom Header & Badges */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    .pill-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-right: 6px;
    }

    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-green { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-purple { background: rgba(192, 132, 252, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }
    .badge-red { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Custom Form & Button Polish */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(56, 189, 248, 0.25);
        background: rgba(30, 41, 59, 0.4);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    /* Expander card */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 10px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "df" not in st.session_state:
    try:
        default_df = load_default_dataset()
        st.session_state.df = default_df.copy()
        st.session_state.original_df = default_df.copy()
        st.session_state.current_dataset_name = "sample_sales.csv"
    except Exception as e:
        st.session_state.df = pd.DataFrame()
        st.session_state.original_df = pd.DataFrame()
        st.session_state.current_dataset_name = "Empty"

if "previous_df" not in st.session_state:
    st.session_state.previous_df = None

if "last_command" not in st.session_state:
    st.session_state.last_command = None

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

if "history" not in st.session_state:
    st.session_state.history = []

if "execution_count" not in st.session_state:
    st.session_state.execution_count = 0

if "input_query_prefill" not in st.session_state:
    st.session_state.input_query_prefill = ""


# Initialize Gemini engine instance
gemini_engine = GeminiEngine()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **Voice2Data AI**")
    st.markdown(
        '<span class="pill-badge badge-blue">Deterministic Engine</span>'
        '<span class="pill-badge badge-green">Zero Code-Gen</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # 1. Dataset Upload Section
    st.markdown("#### 📁 **Dataset Management**")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload a dataset to automate with natural language."
    )

    if uploaded_file is not None:
        new_df, upload_err = load_uploaded_file(uploaded_file)
        if upload_err:
            st.error(upload_err)
        elif new_df is not None and not new_df.empty:
            if st.session_state.current_dataset_name != uploaded_file.name:
                st.session_state.df = new_df.copy()
                st.session_state.original_df = new_df.copy()
                st.session_state.previous_df = None
                st.session_state.current_dataset_name = uploaded_file.name
                st.session_state.history.clear()
                st.session_state.execution_count = 0
                st.session_state.last_result = None
                st.success(f"Loaded `{uploaded_file.name}` ({len(new_df):,} rows)")
                st.rerun()

    # Active dataset metadata
    st.caption(f"**Active Source:** `{st.session_state.current_dataset_name}`")

    # Reset Buttons
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Revert Data", use_container_width=True, help="Revert working data to original loaded state"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.previous_df = None
            st.session_state.last_result = None
            st.session_state.last_command = None
            st.rerun()

    with col_r2:
        if st.button("🗑️ Reset All", use_container_width=True, help="Reset dataset to sample_sales.csv & clear history"):
            default_df = load_default_dataset()
            st.session_state.df = default_df.copy()
            st.session_state.original_df = default_df.copy()
            st.session_state.previous_df = None
            st.session_state.current_dataset_name = "sample_sales.csv"
            st.session_state.history.clear()
            st.session_state.execution_count = 0
            st.session_state.last_result = None
            st.session_state.last_command = None
            st.session_state.input_query_prefill = ""
            st.rerun()

    st.markdown("---")

    # 2. Example Quick-Select Commands
    st.markdown("#### 💡 **Quick Test Commands**")
    example_prompts = [
        "Show top 10 customers by revenue.",
        "Filter Q3 transactions.",
        "Sort products by units sold.",
        "Group revenue by region.",
        "Calculate average revenue by category.",
        "Show electronics sales above 50000.",
        "Select Customer, Revenue and Region columns.",
        "Show bottom 5 transactions by revenue.",
        "Invalid column command.",
        "Unsupported command."
    ]

    for eg in example_prompts:
        if st.button(f"👉 {eg}", key=f"btn_eg_{eg}", use_container_width=True):
            st.session_state.input_query_prefill = eg
            st.rerun()

    st.markdown("---")

    # 3. Supported Operations Cheatsheet
    with st.expander("📖 **Supported Operations**", expanded=False):
        st.markdown("""
        - **Filter / Value Filter**: `Quarter == 'Q3'`, `Revenue > 50000`
        - **Top N / Bottom N**: `Top 10 by Revenue`, `Bottom 5`
        - **Sort**: `Sort by Units descending`
        - **Groupby**: `Group revenue by region`, `Avg revenue by category`
        - **Aggregations**: `Sum`, `Mean`, `Count`, `Min`, `Max`, `Median`
        - **Column Projection**: `Select Customer, Revenue, Region`
        - **Renaming**: `Rename Revenue to Total_Sales`
        - **Date Ranges**: `Filter 2026 transactions`
        """)

    # 4. Execution History Log
    st.markdown("#### 📜 **Execution History**")
    if not st.session_state.history:
        st.caption("No operations executed yet.")
    else:
        for idx, h in enumerate(reversed(st.session_state.history[-8:])):
            status_icon = "🟢" if h["status"] == "SUCCESS" else "🔴"
            st.markdown(
                f"**{status_icon} `{h['timestamp']}`**<br>"
                f"<span style='font-size:0.82rem; color:#cbd5e1;'>Command: *{h['command']}*</span><br>"
                f"<span style='font-size:0.75rem; color:#94a3b8;'>Op: <b>{h['operation']}</b> | Rows Δ: {h['rows_affected']} | Latency: {h['duration_ms']}ms</span>",
                unsafe_allow_html=True
            )
            st.markdown("<hr style='margin: 6px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Page Header & Hero
# ---------------------------------------------------------
st.markdown('<div class="hero-title">⚡ Voice2Data AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Natural Language → Safe Pandas Data Automation Engine</div>', unsafe_allow_html=True)

# Status API Key Indicator
api_key = get_api_key()
if api_key:
    st.markdown(
        '<div style="margin-bottom: 16px;">'
        '<span class="pill-badge badge-green">⚡ Gemini AI Engine: Active</span>'
        '<span class="pill-badge badge-blue">Deterministic Sandbox: Protected</span>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.info("💡 **Demo Mode Active**: Gemini API Key not detected in environment (`.env`). Running in high-performance deterministic fallback mode with full test command support.")


# ---------------------------------------------------------
# KPI Dashboard Section
# ---------------------------------------------------------
kpi_data = compute_kpis(
    current_df=st.session_state.df,
    original_df=st.session_state.original_df,
    previous_df=st.session_state.previous_df,
    execution_count=st.session_state.execution_count
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.metric("Total Rows", f"{kpi_data['rows']:,}", delta=kpi_data['rows_delta'])
with kpi_col2:
    st.metric("Total Columns", f"{kpi_data['cols']:,}", delta=kpi_data['cols_delta'])
with kpi_col3:
    st.metric("Missing Values", f"{kpi_data['missing']:,}", delta=kpi_data['missing_delta'], delta_color="inverse")
with kpi_col4:
    st.metric("Duplicate Rows", f"{kpi_data['duplicates']:,}")
with kpi_col5:
    st.metric("Operations Executed", f"{kpi_data['operations']}", delta=kpi_data['operations_delta'])

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dataset Preview & Schema Inspector
# ---------------------------------------------------------
with st.expander("🔍 **Dataset Schema & Data Quality Profile**", expanded=False):
    quality_profile = get_dataset_quality_profile(st.session_state.df)
    
    q_col1, q_col2, q_col3 = st.columns([1, 1, 2])
    with q_col1:
        st.write(f"**Memory Footprint:** `{quality_profile['memory_usage_kb']} KB`")
    with q_col2:
        st.write(f"**Missing Data Ratio:** `{quality_profile['missing_pct']}%`")
    with q_col3:
        insights = get_quick_insights(st.session_state.df)
        if insights:
            insights_str = " | ".join([f"{k}: **{v}**" for k, v in insights.items()])
            st.markdown(f"**Overview:** {insights_str}")

    st.dataframe(quality_profile["schema_df"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Command Interface Form (Text + Voice)
# ---------------------------------------------------------
st.markdown("### 🤖 **Tell Voice2Data What To Do**")

# Optional Voice Input Widget outside/above the form to capture microphone audio
voice_text = None
with st.expander("🎙️ **Voice Command (Microphone)**", expanded=False):
    st.caption("Record your voice command (e.g. *'Show top 10 customers by revenue'* or *'Filter Q3 transactions'*):")
    audio_value = st.audio_input("Record voice instruction", key="mic_audio_input")
    if audio_value is not None:
        with st.spinner("🎙️ Transcribing voice with Gemini Multimodal AI..."):
            audio_bytes = audio_value.read()
            transcribed_text, trans_err = gemini_engine.transcribe_audio(audio_bytes, mime_type="audio/wav")
            if transcribed_text:
                st.success(f"Transcribed: *\"{transcribed_text}\"*")
                voice_text = transcribed_text
            elif trans_err:
                st.warning(f"Voice note: {trans_err}. You can also type your command below.")

# Main Execution Form
default_query = voice_text if voice_text else st.session_state.input_query_prefill

with st.form(key="command_form", clear_on_submit=False):
    cmd_col1, cmd_col2 = st.columns([5, 1])
    with cmd_col1:
        user_command_input = st.text_input(
            "Natural Language Command:",
            value=default_query,
            placeholder="e.g. Show top 10 customers by revenue, Filter Q3 transactions, Group revenue by region...",
            help="Enter any natural language query or data transformation instruction.",
            label_visibility="collapsed"
        )
    with cmd_col2:
        submit_btn = st.form_submit_button("⚡ Run Engine", use_container_width=True)


# ---------------------------------------------------------
# Core Execution Pipeline
# ---------------------------------------------------------
if submit_btn and user_command_input.strip():
    clean_command = user_command_input.strip()
    st.session_state.input_query_prefill = ""  # Reset prefill
    
    with st.spinner("🧠 Analyzing command & extracting structured intent..."):
        # Step 1: Gemini Intent Extraction
        intent, intent_err = gemini_engine.extract_intent(clean_command, st.session_state.df)

    # Step 2: Deterministic Validator
    is_valid, validated_intent, val_error = validate_intent(intent, st.session_state.df)

    timestamp_str = datetime.now().strftime("%H:%M:%S")

    if not is_valid:
        # Record failed command to history
        st.session_state.history.append({
            "command": clean_command,
            "operation": intent.get("operation", "invalid"),
            "status": "FAILED",
            "timestamp": timestamp_str,
            "rows_affected": 0,
            "duration_ms": 0.0
        })
        st.session_state.last_command = clean_command
        st.session_state.last_intent = intent
        st.session_state.last_result = None

        st.error(f"❌ **Validation Error**: {val_error or 'Command could not be mapped to a valid safe operation.'}")
        if intent.get("operation") == "unsupported":
            st.info("ℹ️ **Tip**: Voice2Data AI supports deterministic data operations (filter, sort, top_n, bottom_n, groupby, aggregate, select_columns, date_filter). Please refer to the supported operations cheatsheet in the sidebar.")

    else:
        # Step 3: Safe Controlled Pandas Engine Execution
        with st.spinner("⚙️ Executing validated intent via Pandas Engine..."):
            st.session_state.previous_df = st.session_state.df.copy()
            exec_result = execute_intent(st.session_state.df, validated_intent)

        if exec_result["success"]:
            # Update state with transformed DataFrame
            st.session_state.df = exec_result["result_df"]
            st.session_state.last_command = clean_command
            st.session_state.last_intent = validated_intent
            st.session_state.last_result = exec_result
            st.session_state.execution_count += 1

            # AI Grounded Explanation
            explanation = gemini_engine.generate_explanation(
                user_command=clean_command,
                intent_summary=exec_result["operation_summary"],
                rows_before=exec_result["rows_before"],
                rows_after=exec_result["rows_after"],
                sample_df=exec_result["result_df"]
            )
            st.session_state.last_explanation = explanation

            # Record success to history
            st.session_state.history.append({
                "command": clean_command,
                "operation": validated_intent.get("operation", "custom"),
                "status": "SUCCESS",
                "timestamp": timestamp_str,
                "rows_affected": exec_result["rows_affected"],
                "duration_ms": exec_result["execution_time_ms"]
            })

            st.rerun()

        else:
            # Execution failure
            st.session_state.history.append({
                "command": clean_command,
                "operation": validated_intent.get("operation", "custom"),
                "status": "FAILED",
                "timestamp": timestamp_str,
                "rows_affected": 0,
                "duration_ms": exec_result["execution_time_ms"]
            })
            st.error(f"❌ **Execution Error**: {exec_result['error']}")


# ---------------------------------------------------------
# Results, Visualizations, and Interactive Data Editor
# ---------------------------------------------------------
if st.session_state.last_result and st.session_state.last_result.get("success"):
    res = st.session_state.last_result

    st.markdown("---")
    st.markdown("### 📊 **Automation Result & Insights**")

    # Execution telemetry pills
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric("Rows Result", f"{res['rows_after']:,}", delta=f"{res['rows_after'] - res['rows_before']:+d}")
    with r_col2:
        st.metric("Rows Affected", f"{res['rows_affected']:,}")
    with r_col3:
        st.metric("Execution Latency", f"{res['execution_time_ms']} ms")
    with r_col4:
        st.metric("Operation", f"{st.session_state.last_intent.get('operation', 'executed').upper()}")

    # AI Grounded Explanation Card
    if st.session_state.last_explanation:
        st.success(f"🤖 **AI Explanation**: {st.session_state.last_explanation}")

    # Security & Intent Transparency
    with st.expander("🛡️ **Inspected Structured Intent (Security Audit)**", expanded=False):
        st.json(st.session_state.last_intent)

    # Visualizations
    st.markdown("#### 📈 **Automated Visualization**")
    fig = create_auto_chart(
        st.session_state.df,
        title_hint=f"Analysis: {st.session_state.last_command}",
        intent=st.session_state.last_intent
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No chart suitable for current dataset structure.")

    st.markdown("---")


# ---------------------------------------------------------
# Interactive Data Editor & Export
# ---------------------------------------------------------
st.markdown("### 📝 **Interactive Data Editor**")
st.caption("You can inspect and directly edit cell values below. Click **'Apply Table Edits'** to persist modifications.")

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
    key="interactive_table_editor"
)

col_ed1, col_ed2, col_ed3 = st.columns([2, 1, 1])
with col_ed1:
    if st.button("💾 Apply Table Edits", use_container_width=True):
        if not edited_df.equals(st.session_state.df):
            st.session_state.previous_df = st.session_state.df.copy()
            st.session_state.df = edited_df.copy()
            st.success("Changes saved successfully into session state!")
            st.rerun()

with col_ed2:
    # CSV Export
    csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"voice2data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_ed3:
    # Excel Export
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        st.session_state.df.to_excel(writer, index=False, sheet_name='Voice2Data_Output')
    st.download_button(
        label="📥 Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"voice2data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
