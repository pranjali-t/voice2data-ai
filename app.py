"""
Voice2Data AI - Natural Language to Safe Pandas Data Automation Engine.
A production-grade Streamlit application for safe, deterministic dataset automation.
Enterprise SaaS Edition.
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
# Vector Lucide SVG Icons (Zero External Dependencies)
# ---------------------------------------------------------
SVG_LOGO = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>"""
SVG_DATABASE = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>"""
SVG_TERMINAL = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>"""
SVG_MIC = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>"""
SVG_SHIELD = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""
SVG_SPARKLES = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>"""
SVG_CHART = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>"""
SVG_TABLE = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/></svg>"""
SVG_CHECK = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
SVG_ALERT = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""


# ---------------------------------------------------------
# Enterprise Dark Theme CSS (Linear / Stripe / Vercel Aesthetic)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base Surface */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #F8FAFC;
    }

    .stApp {
        background-color: #0B0F17;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.35) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(30, 58, 138, 0.18) 0px, transparent 50%);
        color: #F8FAFC;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Surface */
    section[data-testid="stSidebar"] {
        background-color: #0D131F !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.8rem !important;
    }

    /* Metric Cards - Non-Truncating, Baseline Aligned & Responsive */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
        min-height: 104px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.35) !important;
        background: #141E30 !important;
        transform: translateY(-1px);
    }

    /* Fix Metric Label Truncation (Especially Operations Executed) */
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.70rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        display: block !important;
        -webkit-line-clamp: unset !important;
        line-height: 1.25 !important;
        min-height: 2.5em !important;
    }

    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
        line-height: 1.25 !important;
        margin: 0 !important;
        padding: 0 !important;
        -webkit-line-clamp: unset !important;
        color: inherit !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.48rem !important;
        font-weight: 700 !important;
        font-feature-settings: "cv02", "cv03", "cv04", "cv11" !important;
        line-height: 1.2 !important;
        margin-top: 4px !important;
        margin-bottom: 2px !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        line-height: 1 !important;
    }

    @media (max-width: 1100px) {
        div[data-testid="stMetric"] {
            padding: 10px 10px !important;
            min-height: 96px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.65rem !important;
            letter-spacing: 0.02em !important;
        }
    }

    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.15s ease;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: #162032;
        color: #F1F5F9;
    }

    .stButton>button:hover {
        background: #1E293B;
        border-color: rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }

    /* Sidebar Category Headers & Quick Command Buttons */
    .sidebar-category-header {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin: 12px 0 5px 2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.74rem !important;
        padding: 7px 10px !important;
        background: #0E1726 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #CBD5E1 !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        word-break: break-word !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #17243B !important;
        border-color: rgba(59, 130, 246, 0.35) !important;
        color: #60A5FA !important;
        transform: translateX(2px);
    }

    /* Primary Form & Run Button Alignment */
    div[data-testid="stForm"] {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 12px !important;
    }

    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="stForm"] button[kind="formSubmit"] {
        background: #2563EB !important;
        border: 1px solid #3B82F6 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stForm"] button[kind="primary"]:hover,
    div[data-testid="stForm"] button[kind="formSubmit"]:hover {
        background: #1D4ED8 !important;
        border-color: #60A5FA !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4) !important;
    }

    /* Text Inputs */
    div[data-testid="stTextInput"] input {
        background: #0B0F17 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
        padding: 9px 14px !important;
        height: 42px !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        margin-bottom: 12px;
    }

    div[data-testid="stExpander"] details summary {
        font-weight: 500 !important;
        color: #E2E8F0 !important;
        font-size: 0.9rem !important;
    }

    /* Audio Input Widget Styling */
    div[data-testid="stAudioInput"] {
        background: #0B0F17 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }

    /* Dataframe & Editor */
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Custom Header Component */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        padding-bottom: 18px;
        margin-bottom: 24px;
    }

    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.15);
        flex-shrink: 0;
    }

    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .edition-badge {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 2px 7px;
        border-radius: 4px;
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .brand-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    .status-group {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 11px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        line-height: 1;
    }

    .status-badge svg {
        flex-shrink: 0;
    }

    .status-green {
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .status-blue {
        background: rgba(59, 130, 246, 0.1);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .status-amber {
        background: rgba(245, 158, 11, 0.1);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
        flex-shrink: 0;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.02rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 4px;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }

    .section-header svg {
        flex-shrink: 0;
    }

    /* Sidebar Metadata Card */
    .sidebar-meta-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 6px;
        padding: 10px 12px;
        margin: 10px 0 14px 0;
    }

    .sidebar-meta-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
    }

    .sidebar-meta-val {
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        color: #F1F5F9;
        word-break: break-all;
    }

    /* Voice Command HUD & Animated Waveform */
    .voice-hud-container {
        background: #0E1626;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .voice-hud-info {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .voice-hud-icon-wrap {
        width: 32px;
        height: 32px;
        border-radius: 6px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #60A5FA;
        flex-shrink: 0;
    }

    .voice-hud-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #F8FAFC;
    }

    .voice-hud-sub {
        font-size: 0.73rem;
        color: #94A3B8;
        margin-top: 1px;
    }

    .voice-visualizer-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .waveform-visualizer {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 22px;
        padding: 0 4px;
    }

    .waveform-bar {
        width: 3px;
        background: #3B82F6;
        border-radius: 2px;
        animation: waveformPulse 1.4s ease-in-out infinite;
    }

    .waveform-bar:nth-child(1) { height: 6px; animation-delay: 0.1s; }
    .waveform-bar:nth-child(2) { height: 14px; animation-delay: 0.25s; }
    .waveform-bar:nth-child(3) { height: 20px; animation-delay: 0.4s; background: #60A5FA; }
    .waveform-bar:nth-child(4) { height: 15px; animation-delay: 0.15s; }
    .waveform-bar:nth-child(5) { height: 22px; animation-delay: 0.35s; background: #818CF8; }
    .waveform-bar:nth-child(6) { height: 11px; animation-delay: 0.2s; }
    .waveform-bar:nth-child(7) { height: 6px; animation-delay: 0.05s; }

    @keyframes waveformPulse {
        0%, 100% {
            transform: scaleY(0.4);
            opacity: 0.4;
        }
        50% {
            transform: scaleY(1);
            opacity: 1;
        }
    }

    .voice-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .voice-status-ready {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Voice Cards */
    .voice-transcription-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
    }

    .voice-card-success {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .voice-card-error {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .voice-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }

    .voice-card-badge-success {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #34D399;
    }

    .voice-card-badge-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #FBBF24;
    }

    .voice-card-model {
        font-size: 0.68rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }

    .voice-card-quote {
        font-size: 0.95rem;
        font-weight: 500;
        color: #F8FAFC;
        font-style: italic;
        margin-bottom: 6px;
    }

    .voice-err-text {
        font-size: 0.84rem;
        color: #FDE68A;
        margin-bottom: 6px;
    }

    .voice-card-footer {
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 6px;
    }

    .voice-card-hint {
        font-size: 0.74rem;
        color: #94A3B8;
    }

    /* History Entry */
    .history-entry {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        transition: border-color 0.15s ease;
    }

    .history-entry:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    .history-row-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        margin-bottom: 4px;
    }

    .history-cmd {
        font-size: 0.8rem;
        color: #E2E8F0;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .history-telemetry {
        font-size: 0.7rem;
        color: #64748B;
        margin-top: 3px;
        display: flex;
        gap: 8px;
    }

    /* AI Explanation Box */
    .ai-summary-box {
        background: #0F172A;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 14px 0;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }

    .ai-summary-text {
        font-size: 0.88rem;
        color: #E2E8F0;
        line-height: 1.5;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization (Preserved 100%)
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

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None

if "last_transcription_error" not in st.session_state:
    st.session_state.last_transcription_error = None


# Initialize Gemini engine instance
gemini_engine = GeminiEngine()


# ---------------------------------------------------------
# Sidebar - Dataset Management & Prompt Shortcuts
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">'
        f'{SVG_DATABASE} <span style="font-size:0.88rem; font-weight:700; color:#F8FAFC; letter-spacing:0.04em;">DATASET MANAGEMENT</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 1. Dataset Upload Section
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
                st.session_state.last_audio_hash = None
                st.session_state.last_transcription = None
                st.session_state.last_transcription_error = None
                st.success(f"Loaded `{uploaded_file.name}` ({len(new_df):,} rows)")
                st.rerun()

    # Active dataset metadata card
    active_name = st.session_state.current_dataset_name
    row_count = len(st.session_state.df) if not st.session_state.df.empty else 0
    col_count = len(st.session_state.df.columns) if not st.session_state.df.empty else 0

    st.markdown(
        f'<div class="sidebar-meta-card">'
        f'<div class="sidebar-meta-title">Active Source</div>'
        f'<div class="sidebar-meta-val">{active_name}</div>'
        f'<div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">{row_count:,} rows · {col_count} columns</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Reset Buttons
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Revert Data", use_container_width=True, help="Revert working data to original loaded state"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.previous_df = None
            st.session_state.last_result = None
            st.session_state.last_command = None
            st.rerun()

    with col_r2:
        if st.button("Reset Session", use_container_width=True, help="Reset dataset to sample_sales.csv & clear history"):
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
            st.session_state.last_audio_hash = None
            st.session_state.last_transcription = None
            st.session_state.last_transcription_error = None
            st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 16px 0;'>", unsafe_allow_html=True)

    # 2. Example Quick-Select Commands
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
        f'{SVG_TERMINAL} <span style="font-size:0.85rem; font-weight:700; color:#F8FAFC; letter-spacing:0.04em;">QUICK TEST COMMANDS</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    command_groups = [
        (
            "Rankings & Aggregations",
            [
                "Show top 10 customers by revenue.",
                "Show bottom 5 transactions by revenue.",
                "Group revenue by region.",
                "Calculate average revenue by category."
            ]
        ),
        (
            "Filters & Transforms",
            [
                "Filter Q3 transactions.",
                "Show electronics sales above 50000.",
                "Sort products by units sold.",
                "Select Customer, Revenue and Region columns."
            ]
        ),
        (
            "Validation & Edge Tests",
            [
                "Invalid column command.",
                "Unsupported command."
            ]
        )
    ]

    for group_title, commands in command_groups:
        st.markdown(f'<div class="sidebar-category-header">{group_title}</div>', unsafe_allow_html=True)
        for eg in commands:
            if st.button(f"›  {eg}", key=f"btn_eg_{eg}", use_container_width=True):
                st.session_state.input_query_prefill = eg
                st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 16px 0;'>", unsafe_allow_html=True)

    # 3. Supported Operations Cheatsheet
    with st.expander("Supported Capabilities", expanded=False):
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
    st.markdown(
        f'<div style="font-size:0.82rem; font-weight:700; color:#F8FAFC; letter-spacing:0.04em; margin: 16px 0 10px 0;">AUDIT HISTORY</div>',
        unsafe_allow_html=True
    )
    if not st.session_state.history:
        st.caption("No operations executed in current session.")
    else:
        for idx, h in enumerate(reversed(st.session_state.history[-8:])):
            status_color = "#10B981" if h["status"] == "SUCCESS" else "#EF4444"
            st.markdown(
                f'<div class="history-entry">'
                f'<div class="history-row-top">'
                f'<span style="display:flex; align-items:center; gap:5px;">'
                f'<span style="width:6px; height:6px; border-radius:50%; background-color:{status_color};"></span>'
                f'<span style="color:#64748B; font-family:\'JetBrains Mono\',monospace;">{h["timestamp"]}</span>'
                f'</span>'
                f'<span class="edition-badge" style="font-size:0.62rem;">{h["operation"].upper()}</span>'
                f'</div>'
                f'<div class="history-cmd" title="{h["command"]}">{h["command"]}</div>'
                f'<div class="history-telemetry">'
                f'<span>Δ {h["rows_affected"]} rows</span>'
                f'<span>·</span>'
                f'<span>{h["duration_ms"]:.1f}ms</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ---------------------------------------------------------
# Main Page Header & Real-Time Status Bar
# ---------------------------------------------------------
api_key = get_api_key()
engine_status_html = (
    '<div class="status-badge status-green"><span class="pulse-dot"></span><span>Gemini AI: Connected</span></div>'
    if api_key else
    '<div class="status-badge status-amber"><span>Demo Fallback Mode</span></div>'
)

st.markdown(
    f'<div class="top-nav">'
    f'<div class="brand-group">'
    f'<div class="brand-icon">{SVG_LOGO}</div>'
    f'<div>'
    f'<div class="brand-title">'
    f'<span>Voice2Data AI</span>'
    f'<span class="edition-badge">ENTERPRISE v2.4</span>'
    f'</div>'
    f'<div class="brand-subtitle">Natural Language & Voice → Controlled Pandas Automation Engine</div>'
    f'</div>'
    f'</div>'
    f'<div class="status-group">'
    f'{engine_status_html}'
    f'<div class="status-badge status-blue">{SVG_SHIELD} <span>Deterministic Sandbox Enforced</span></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)


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

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dataset Preview & Schema Inspector
# ---------------------------------------------------------
with st.expander("Dataset Schema & Data Quality Profile", expanded=False):
    quality_profile = get_dataset_quality_profile(st.session_state.df)
    
    q_col1, q_col2, q_col3 = st.columns([1, 1, 2])
    with q_col1:
        st.markdown(f"**Memory Footprint:** `{quality_profile['memory_usage_kb']} KB`")
    with q_col2:
        st.markdown(f"**Missing Ratio:** `{quality_profile['missing_pct']}%`")
    with q_col3:
        insights = get_quick_insights(st.session_state.df)
        if insights:
            insights_str = " · ".join([f"{k}: **{v}**" for k, v in insights.items()])
            st.markdown(f"**Overview:** {insights_str}")

    st.dataframe(quality_profile["schema_df"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Command Interface Form (Text + Voice)
# ---------------------------------------------------------
st.markdown(
    f'<div class="section-header">{SVG_TERMINAL} <span>Command Execution Center</span></div>',
    unsafe_allow_html=True
)

# Voice Input Widget outside the form to capture microphone audio
voice_active = (
    st.session_state.get("last_transcription") is not None
    or st.session_state.get("last_transcription_error") is not None
)

with st.expander("Voice Command (Microphone)", expanded=voice_active):
    # Professional Voice HUD with subtle animated waveform
    st.markdown(
        f'<div class="voice-hud-container">'
        f'<div class="voice-hud-info">'
        f'<div class="voice-hud-icon-wrap">{SVG_MIC}</div>'
        f'<div>'
        f'<div class="voice-hud-label">Multimodal Voice Command Interface</div>'
        f'<div class="voice-hud-sub">Speak natural language instructions · Auto-detected MIME · Real-time Gemini STT</div>'
        f'</div>'
        f'</div>'
        f'<div class="voice-visualizer-wrap">'
        f'<div class="waveform-visualizer">'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'<div class="waveform-bar"></div>'
        f'</div>'
        f'<span class="voice-status-pill voice-status-ready">Ready</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    audio_value = st.audio_input("Record voice instruction", key="mic_audio_input")

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        if audio_bytes:
            audio_hash = hash(audio_bytes)
            # Only invoke Gemini when a new audio recording is captured
            if audio_hash != st.session_state.last_audio_hash:
                with st.spinner("Processing audio with Gemini Multimodal STT..."):
                    detected_mime = getattr(audio_value, "type", None) or "audio/wav"
                    transcribed_text, trans_err = gemini_engine.transcribe_audio(
                        audio_bytes, mime_type=detected_mime
                    )
                    st.session_state.last_audio_hash = audio_hash
                    st.session_state.last_transcription = transcribed_text
                    st.session_state.last_transcription_error = trans_err
                    if transcribed_text:
                        st.session_state.input_query_prefill = transcribed_text
                        st.rerun()

            if st.session_state.last_transcription:
                st.markdown(
                    f'<div class="voice-transcription-card voice-card-success">'
                    f'<div class="voice-card-header">'
                    f'<div class="voice-card-badge-success">{SVG_CHECK} <span>Voice Command Recognized</span></div>'
                    f'<span class="voice-card-model">Gemini Multimodal STT</span>'
                    f'</div>'
                    f'<div class="voice-card-quote">"{st.session_state.last_transcription}"</div>'
                    f'<div class="voice-card-footer">'
                    f'<span class="voice-card-hint">Command transferred to query input below. Click <b>Run Analysis</b> to execute.</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif st.session_state.last_transcription_error:
                st.markdown(
                    f'<div class="voice-transcription-card voice-card-error">'
                    f'<div class="voice-card-header">'
                    f'<div class="voice-card-badge-warn">{SVG_ALERT} <span>Voice Transcription Note</span></div>'
                    f'</div>'
                    f'<div class="voice-err-text">{st.session_state.last_transcription_error}</div>'
                    f'<div class="voice-card-footer">'
                    f'<span class="voice-card-hint">You can also type or paste your command directly into the input box below.</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        # Reset audio cache when recording is cleared
        if st.session_state.last_audio_hash is not None:
            st.session_state.last_audio_hash = None
            st.session_state.last_transcription = None
            st.session_state.last_transcription_error = None

# Main Execution Form
default_query = st.session_state.input_query_prefill

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
        submit_btn = st.form_submit_button("Run Analysis", use_container_width=True)


# ---------------------------------------------------------
# Core Execution Pipeline (Preserved 100%)
# ---------------------------------------------------------
if submit_btn and user_command_input.strip():
    clean_command = user_command_input.strip()
    st.session_state.input_query_prefill = ""  # Reset prefill
    st.session_state.last_transcription = None
    st.session_state.last_transcription_error = None
    
    with st.spinner("Extracting structured intent via Gemini AI..."):
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

        st.error(f"Validation Error: {val_error or 'Command could not be mapped to a valid safe operation.'}")
        if intent.get("operation") == "unsupported":
            st.info("Supported Operations: Voice2Data AI supports deterministic operations (filter, sort, top_n, bottom_n, groupby, aggregate, select_columns, date_filter). Please refer to the capabilities cheatsheet in the sidebar.")

    else:
        # Step 3: Safe Controlled Pandas Engine Execution
        with st.spinner("Executing validated intent via Controlled Pandas Engine..."):
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
            st.error(f"Execution Error: {exec_result['error']}")


# ---------------------------------------------------------
# Results, Visualizations, and Interactive Data Editor
# ---------------------------------------------------------
if st.session_state.last_result and st.session_state.last_result.get("success"):
    res = st.session_state.last_result

    st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-header">{SVG_CHART} <span>Automation Results & Insights</span></div>',
        unsafe_allow_html=True
    )

    # Execution telemetry metrics
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
        st.markdown(
            f'<div class="ai-summary-box">'
            f'<div style="flex-shrink:0; margin-top:2px;">{SVG_SPARKLES}</div>'
            f'<div>'
            f'<div style="font-size:0.72rem; font-weight:700; color:#818CF8; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px;">Grounded AI Analysis Summary</div>'
            f'<div class="ai-summary-text">{st.session_state.last_explanation}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Security & Intent Transparency
    with st.expander("Security Audit & Structured Intent Verification", expanded=False):
        sec_col1, sec_col2, sec_col3 = st.columns(3)
        with sec_col1:
            st.markdown(
                f'<div class="status-badge status-green" style="width:100%; justify-content:center;">'
                f'{SVG_CHECK} <span>Deterministic Whitelist: PASSED</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with sec_col2:
            st.markdown(
                f'<div class="status-badge status-green" style="width:100%; justify-content:center;">'
                f'{SVG_CHECK} <span>Code Injection Scan: ZERO CODE DETECTED</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with sec_col3:
            st.markdown(
                f'<div class="status-badge status-blue" style="width:100%; justify-content:center;">'
                f'{SVG_SHIELD} <span>Target Columns Whitelist: ENFORCED</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.json(st.session_state.last_intent)

    # Visualizations
    st.markdown(
        f'<div class="section-header" style="margin-top:20px;">{SVG_CHART} <span>Automated Visualization</span></div>',
        unsafe_allow_html=True
    )
    fig = create_auto_chart(
        st.session_state.df,
        title_hint=f"Analysis: {st.session_state.last_command}",
        intent=st.session_state.last_intent
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No chart suitable for current dataset structure.")


# ---------------------------------------------------------
# Interactive Data Editor & Export Hub
# ---------------------------------------------------------
st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown(
    f'<div class="section-header">{SVG_TABLE} <span>Interactive Dataset View & Export</span></div>',
    unsafe_allow_html=True
)
st.caption("Inspect and directly edit cell values below. Click 'Apply Table Edits' to persist modifications.")

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
    key="interactive_table_editor"
)

col_ed1, col_ed2, col_ed3 = st.columns([2, 1, 1])
with col_ed1:
    if st.button("Apply Table Edits", use_container_width=True):
        if not edited_df.equals(st.session_state.df):
            st.session_state.previous_df = st.session_state.df.copy()
            st.session_state.df = edited_df.copy()
            st.success("Changes saved successfully into session state!")
            st.rerun()

with col_ed2:
    csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"voice2data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_ed3:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        st.session_state.df.to_excel(writer, index=False, sheet_name='Voice2Data_Output')
    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"voice2data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
