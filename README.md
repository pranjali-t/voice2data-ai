## 🚀 Live Demo

[Open Voice2Data AI](https://voice2data-ai-n5wwrqtxuylyddnkpsa97o.streamlit.app/)


 # ⚡ VOICE2DATA AI
> **Natural Language → Safe Pandas Data Automation Engine**  
> *A Production-Grade Generative AI & Data Analytics System with Zero Code-Generation Risk.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API%202.5-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Problem Statement
Traditional data analytics interfaces require either deep SQL/Pandas programming skills or complex BI software with steep learning curves. While Large Language Models (LLMs) can bridge this gap by converting natural language into code, **directly executing LLM-generated code (`eval()`, `exec()`) introduces catastrophic security risks**, including arbitrary code execution, unintended data corruption, memory leaks, and hallucinated mathematical results.

## 🎯 2. Motivation
Modern enterprise analytics platforms demand the convenience of conversational AI paired with **deterministic mathematical guarantees and airtight security boundaries**. **Voice2Data AI** was built to solve this dilemma: allowing business stakeholders and data scientists to control tabular datasets via natural language voice and text commands without ever running unverified code.

## 💡 3. Proposed Solution
Voice2Data AI implements a **multi-tier sandboxed architecture**:
1. **Gemini** functions strictly as an **Intent Extraction Engine**, transforming unstructured voice/text into structured JSON schema payloads.
2. A **Deterministic Validator** verifies that the requested operation, referenced columns, and parameters strictly conform to allowed whitelists and actual dataset schemas.
3. A **Controlled Pandas Engine** executes only hardcoded, explicit Pandas API routines (no `eval()` or `exec()`).
4. A **Modern SaaS Streamlit Dashboard** visualizes the computed results, tracks execution telemetry, generates interactive Plotly charts, and provides an editable data grid.

---

## 🛡️ 4. Core Security & Architectural Principle

```
┌─────────────────────────────────────────────────────────────┐
│                       VOICE2DATA AI                         │
│                                                             │
│   GEMINI      ──►   VALIDATOR   ──►   PANDAS   ──► STREAMLIT│
│  (Understand)        (Protect)       (Execute)   (Visualize)│
└─────────────────────────────────────────────────────────────┘
```

- **Gemini = Understand**: Translates natural language into structured JSON intent.
- **Validator = Protect**: Guards against non-existent columns, out-of-bounds limits, and injection payloads.
- **Pandas = Execute**: Performs deterministic calculations (source of mathematical truth).
- **Streamlit = Visualize**: Displays real-time KPIs, interactive data editors, and dynamic charts.

---

## 🏗️ 5. System Architecture & Flow

```mermaid
flowchart TD
    User([👤 User]) --> UI[⚡ Streamlit Dark SaaS Dashboard]
    UI --> InputChoice{Input Method}
    InputChoice -->|Typed Command| Pipeline[Command Processor]
    InputChoice -->|🎙️ Mic Audio| STT[Gemini Multimodal Speech-to-Text] --> Pipeline

    subgraph Security Boundary
        Pipeline --> Gemini[🧠 Gemini Intent Engine]
        Gemini -->|Strict Structured JSON| Validator{🛡️ Deterministic Validator}
        Validator -->|Invalid / Code Injection / Missing Col| ErrorHandler[❌ Safe Friendly Error & Audit Log]
        Validator -->|Valid Schema & Whitelisted Op| PandasEngine[⚙️ Controlled Pandas Engine]
    end

    PandasEngine -->|Calculated DataFrame + Stats| ResultProcessor[📊 Result & Telemetry Processor]
    ResultProcessor --> Metrics[KPI Dashboard & Metric Deltas]
    ResultProcessor --> Editor[📝 Interactive Data Editor]
    ResultProcessor --> Charts[📈 Automated Smart Plotly Visualizations]
    ResultProcessor --> History[📜 Execution History & Session State]
    ResultProcessor --> AIExplain[🤖 AI Grounded Transformation Summary]

    ErrorHandler --> History
```

---

## ✨ 6. Key Features

- 🎙️ **Dual-Mode Input (Voice & Text)**: Native microphone voice recording with Gemini Multimodal transcription alongside form-based typed queries.
- 🛡️ **Zero Code-Gen Security**: Absolute guarantee against arbitrary code execution (`eval()`/`exec()` are completely omitted).
- 📊 **Real-Time KPI Dashboard**: Interactive metrics displaying total rows, columns, missing values, duplicates, and execution deltas.
- 🔍 **Dataset Schema & Quality Profiler**: Automated memory profiling, data type inspection, null ratios, and unique value analysis.
- ⚙️ **High-Performance Pandas Execution**: Sub-10ms latency for deterministic data transformations on 1,000+ row datasets.
- 📝 **Interactive Data Editor**: In-place table cell editing with one-click state persistence (`st.data_editor`).
- 📈 **Intelligent Auto-Visualization**: Dynamic Plotly charts (Time Series, Donut, Ranked Horizontal Bar, Multi-variate Scatter) styled with a dark SaaS aesthetic.
- 📜 **Session History & Telemetry**: Full chronological audit trail with success/failure status badges, timestamps, and row delta metrics.
- 💾 **Dual-Format Export**: One-click download of transformed data to CSV and Microsoft Excel (`.xlsx`).
- 🚀 **Offline Resilience Fallback**: Deterministic intent parsing fallback when running in offline or demo environments without an API key.

---

## 📖 7. Supported Operations

| Operation | JSON Action | Description & Example |
| :--- | :--- | :--- |
| **Filter / Value Filter** | `filter` | Row boolean filtering (`"Filter Q3 transactions"`, `"Revenue > 50000"`) |
| **Top N / Bottom N** | `top_n`, `bottom_n` | Ranked slices (`"Show top 10 customers by revenue"`, `"Bottom 5"`) |
| **Sort** | `sort` | Single or multi-column ordering (`"Sort products by units sold"`) |
| **Groupby & Aggregations**| `groupby` | Categorical grouping with `sum`, `mean`, `count`, `min`, `max` (`"Group revenue by region"`) |
| **Dataset Aggregations** | `aggregate` | Whole-dataset metrics calculation |
| **Column Projection** | `select_columns` | Select subset of columns (`"Select Customer, Revenue and Region"`) |
| **Column Renaming** | `rename_columns` | Rename columns (`"Rename Revenue to Total_Sales"`) |
| **Date Range Filter** | `date_filter` | Temporal filtering by date, quarter, year, or month range |
| **Composite Operations** | `filter_sort_top` | Multi-step pipeline (`"Filter Q3 transactions and show top 10 by revenue"`) |
| **Graceful Rejection** | `unsupported` | Safe error handling for out-of-scope or malicious queries |

---

## 🛠️ 8. Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io) (Dark SaaS theme, session state management, native audio input)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org) (Controlled vectorized operations)
- **Generative AI Engine**: [Google Gemini API](https://ai.google.dev/) (`google-genai` & `google-generativeai` SDKs)
- **Data Visualization**: [Plotly](https://plotly.com) (Interactive dark-themed charts)
- **Spreadsheet Engine**: [openpyxl](https://openpyxl.readthedocs.io) (Excel `.xlsx` read/write)
- **Environment Config**: [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 📂 9. Project Structure

```
Voice2Data-AI/
├── app.py                     # Streamlit SaaS dashboard & UI application
│
├── engine/
│   ├── __init__.py            # Engine package initializer
│   ├── gemini_engine.py       # Gemini API client, intent extractor & audio transcription
│   ├── validator.py           # Deterministic schema, type, column & security validator
│   ├── pandas_engine.py       # Safe, controlled Pandas execution engine (no eval/exec)
│   └── prompts.py             # System prompt, dynamic prompt builder & JSON schemas
│
├── utils/
│   ├── __init__.py            # Utilities package initializer
│   ├── data_loader.py         # CSV/Excel parser, default dataset loader & schema profiler
│   ├── analytics.py           # KPI metrics, deltas & summary insights
│   └── visualization.py       # Intelligent dark-themed Plotly charts generator
│
├── data/
│   └── sample_sales.csv       # Default demo sales dataset (1,000 transactions, 12 columns)
│
├── test_suite.py              # Automated test suite covering all 10 core capstone commands
├── requirements.txt           # Clean production dependencies for Streamlit Cloud
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Standard Python & environment ignore rules
└── .env.example               # Template for environment variables
```

---

## 🚀 10. Local Setup & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Voice2Data-AI.git
cd Voice2Data-AI
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your Google Gemini API Key:
```ini
GEMINI_API_KEY=AIzaSyYourActualGeminiAPIKeyHere
```
*(Note: If no API key is provided, the application automatically runs in deterministic demo fallback mode with full test command support).*

### Step 5: Run Automated Tests
```bash
python test_suite.py
```

### Step 6: Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 11. Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub repository.
3. Set the Main file path to `app.py`.
4. In **Advanced Settings** → **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy**. Voice2Data AI is completely software-only and ready for production!

---

## 🧪 12. Verification & Example Commands

Test the system using these 10 standard commands:

1. `Show top 10 customers by revenue.`
2. `Filter Q3 transactions.`
3. `Sort products by units sold.`
4. `Group revenue by region.`
5. `Calculate average revenue by category.`
6. `Show electronics sales above 50000.`
7. `Select Customer, Revenue and Region columns.`
8. `Show bottom 5 transactions by revenue.`
9. `Invalid column command.` *(Safely rejected by Validator)*
10. `Unsupported command.` *(Safely handled without crashing)*

---

## 🔮 13. Future Scope
- Multi-dataset joins and relationship graphs across multiple CSV files.
- Exporting automated visual analytics reports to PDF.
- Dynamic pivot tables with drag-and-drop hierarchy adjustments.
- Integration with cloud data warehouses (Snowflake, BigQuery) using deterministic query builders.

---

## 👤 Author & Capstone Information
- **Project**: Voice2Data AI - B.Tech Capstone Project
- **Domain**: Generative AI, Data Engineering & Secure Systems
- **Core Technologies**: Python, Streamlit, Pandas, Google Gemini API
