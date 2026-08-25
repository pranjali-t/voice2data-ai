"""
Gemini Intent Engine for Voice2Data AI.
Interfaces safely with Google Gemini API to extract structured JSON intents
and generate grounded natural language explanations.
"""
import os
import re
import json
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from dotenv import load_dotenv

# Resolve project root and possible env file paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
ENV_TXT_FILE = os.path.join(PROJECT_ROOT, ".env.txt")

def _load_all_dotenv():
    """Loads environment variables safely from .env or .env.txt."""
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    elif os.path.exists(ENV_TXT_FILE):
        load_dotenv(ENV_TXT_FILE, override=True)
    else:
        load_dotenv(override=True)

_load_all_dotenv()

from engine.prompts import (
    SYSTEM_PROMPT,
    build_intent_extraction_prompt,
    build_explanation_prompt
)

# Optional Streamlit import for secrets
try:
    import streamlit as st
except ImportError:
    st = None


def get_api_key() -> Optional[str]:
    """Retrieves and sanitizes the Gemini API Key from Streamlit secrets, environment, or .env."""
    _load_all_dotenv()

    # 1. Streamlit Secrets
    if st is not None:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
                if key and key != "your_gemini_api_key_here":
                    return key
        except Exception:
            pass
    
    # 2. Environment Variables
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        clean_key = env_key.strip().strip('"').strip("'")
        if clean_key and clean_key != "your_gemini_api_key_here":
            return clean_key
        
    return None


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON from Gemini's response, handling code fences or extra text."""
    text = raw_text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Find matching curly braces if extra text surrounds JSON
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx:end_idx + 1]

    # Handle trailing commas which are invalid in JSON
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return json.loads(text)


def rule_based_fallback_intent(user_command: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligent heuristic fallback for common test queries when API key is missing or offline.
    Ensures maximum resilience and zero-crash capability.
    """
    cmd = user_command.strip().lower()
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # 1. Top N / Bottom N
    top_match = re.search(r"top\s+(\d+)\s*(?:by\s+([a-zA-Z_]+)|customers?\s+by\s+([a-zA-Z_]+)|products?\s+by\s+([a-zA-Z_]+))?", cmd)
    bottom_match = re.search(r"bottom\s+(\d+)\s*(?:by\s+([a-zA-Z_]+)|transactions?\s+by\s+([a-zA-Z_]+))?", cmd)

    if top_match:
        n = int(top_match.group(1))
        target_col = None
        for group_idx in (2, 3, 4):
            if top_match.group(group_idx):
                c_name = top_match.group(group_idx).lower()
                if c_name in cols_lower:
                    target_col = cols_lower[c_name]
        if not target_col:
            for num_col in ["Revenue", "Units", "Unit_Price"]:
                if num_col in cols:
                    target_col = num_col
                    break

        if "customer" in cmd and "Customer" in cols:
            return {
                "operation": "top_n",
                "column": target_col or "Revenue",
                "n": n,
                "ascending": False,
                "group_by": "Customer",
                "agg_func": "sum"
            }
        return {
            "operation": "top_n",
            "column": target_col or cols[0],
            "n": n,
            "ascending": False
        }

    if bottom_match:
        n = int(bottom_match.group(1))
        target_col = None
        for group_idx in (2, 3):
            if bottom_match.group(group_idx):
                c_name = bottom_match.group(group_idx).lower()
                if c_name in cols_lower:
                    target_col = cols_lower[c_name]
        if not target_col:
            for num_col in ["Revenue", "Units", "Unit_Price"]:
                if num_col in cols:
                    target_col = num_col
                    break
        return {
            "operation": "bottom_n",
            "column": target_col or cols[0],
            "n": n,
            "ascending": True
        }

    # 2. Filter Q3 / Quarter
    quarter_match = re.search(r"\b(q[1-4])\b", cmd)
    if quarter_match and "Quarter" in cols:
        q_val = quarter_match.group(1).upper()
        return {
            "operation": "filter",
            "conditions": [{"column": "Quarter", "operator": "==", "value": q_val}],
            "logical_operator": "AND"
        }

    # 3. Filter category / value (e.g. "Show electronics sales above 50000")
    if "above" in cmd or "greater than" in cmd or ">" in cmd:
        num_val_match = re.search(r"(?:above|greater than|>)\s*(\d+)", cmd)
        conditions = []
        if "Category" in cols and "electronics" in cmd:
            conditions.append({"column": "Category", "operator": "==", "value": "Electronics"})
        if num_val_match and "Revenue" in cols:
            conditions.append({"column": "Revenue", "operator": ">", "value": float(num_val_match.group(1))})
        if conditions:
            return {
                "operation": "filter",
                "conditions": conditions,
                "logical_operator": "AND"
            }

    # 4. Sort
    if "sort" in cmd:
        if "units" in cmd and "Units" in cols:
            return {"operation": "sort", "sort_by": [{"column": "Units", "ascending": False}]}
        if "revenue" in cmd and "Revenue" in cols:
            return {"operation": "sort", "sort_by": [{"column": "Revenue", "ascending": False}]}
        if "date" in cmd and "Date" in cols:
            return {"operation": "sort", "sort_by": [{"column": "Date", "ascending": True}]}

    # 5. Groupby / Aggregate
    if "group" in cmd or "by region" in cmd or "by category" in cmd:
        if "region" in cmd and "Region" in cols:
            return {
                "operation": "groupby",
                "group_by": ["Region"],
                "aggregations": [{"column": "Revenue" if "Revenue" in cols else cols[0], "agg_func": "sum"}],
                "sort_column": "Revenue" if "Revenue" in cols else None,
                "ascending": False
            }
        if "category" in cmd and "Category" in cols:
            agg_func = "mean" if "average" in cmd or "avg" in cmd or "mean" in cmd else "sum"
            return {
                "operation": "groupby",
                "group_by": ["Category"],
                "aggregations": [{"column": "Revenue" if "Revenue" in cols else cols[0], "agg_func": agg_func}],
                "sort_column": "Revenue" if "Revenue" in cols else None,
                "ascending": False
            }

    # 6. Average revenue by category
    if "average" in cmd or "avg" in cmd or "mean" in cmd:
        if "category" in cmd and "Category" in cols:
            return {
                "operation": "groupby",
                "group_by": ["Category"],
                "aggregations": [{"column": "Revenue" if "Revenue" in cols else cols[0], "agg_func": "mean"}],
                "sort_column": "Revenue" if "Revenue" in cols else None,
                "ascending": False
            }

    # 7. Select columns (ordered by appearance in command)
    if "select" in cmd or "columns" in cmd:
        selected_with_pos = []
        for c in cols:
            pos = cmd.find(c.lower())
            if pos != -1:
                selected_with_pos.append((pos, c))
        if selected_with_pos:
            selected_with_pos.sort(key=lambda x: x[0])
            return {
                "operation": "select_columns",
                "columns": [c for _, c in selected_with_pos]
            }

    return {
        "operation": "unsupported",
        "reason": "Could not understand command. Please verify your query or check available operations in the sidebar."
    }


class GeminiEngine:
    """Core interface for Gemini API intent extraction and multimodal processing."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key()
        self._client = None
        self._genai = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini API client safely."""
        current_key = self.api_key or get_api_key()
        if not current_key:
            return

        self.api_key = current_key

        # Attempt modern google-genai SDK first
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return
        except Exception:
            pass

        # Fallback to google-generativeai SDK
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._genai = genai_legacy
        except Exception:
            pass

    def extract_intent(self, user_command: str, df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Sends the user command and dataset schema context to Gemini to extract a structured intent JSON.
        
        Returns:
            Tuple of (intent_dict, error_message)
        """
        if not self._client and not self._genai:
            self._init_client()

        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_data = df.head(3).to_string(index=False)

        prompt = build_intent_extraction_prompt(
            user_command=user_command,
            columns=columns,
            dtypes=dtypes,
            sample_data=sample_data
        )

        if not self.api_key or (not self._client and not self._genai):
            # Use intelligent rule fallback if API key is not configured
            fallback_intent = rule_based_fallback_intent(user_command, df)
            return fallback_intent, None

        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

        # Try google-genai client
        if self._client:
            for model_name in models_to_try:
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "system_instruction": SYSTEM_PROMPT,
                            "response_mime_type": "application/json",
                            "temperature": 0.0
                        }
                    )
                    if response and response.text:
                        parsed = clean_json_response(response.text)
                        return parsed, None
                except Exception:
                    continue

        # Try legacy google-generativeai client
        if self._genai:
            for model_name in models_to_try:
                try:
                    model = self._genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_PROMPT,
                        generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
                    )
                    response = model.generate_content(prompt)
                    if response and response.text:
                        parsed = clean_json_response(response.text)
                        return parsed, None
                except Exception:
                    continue

        # If all API calls fail, fallback gracefully
        fallback_intent = rule_based_fallback_intent(user_command, df)
        return fallback_intent, "Gemini API unavailable or quota reached; utilized deterministic intent parser."

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> Tuple[Optional[str], Optional[str]]:
        """
        Transcribes speech audio into text using Gemini's native multimodal capabilities.
        
        Returns:
            Tuple of (transcribed_text, error_message)
        """
        if not audio_bytes:
            return None, "No audio data received."

        if not self._client and not self._genai:
            self._init_client()

        if not self.api_key:
            return None, "Gemini API key required for voice transcription."

        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        prompt = "Transcribe the spoken words in this audio exactly. Return ONLY the transcribed text without extra remarks."

        if self._client:
            from google.genai import types
            for model_name in models_to_try:
                try:
                    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=[part, prompt]
                    )
                    if response and response.text:
                        return response.text.strip().strip('"'), None
                except Exception as e:
                    last_err = str(e)
                    continue

        if self._genai:
            for model_name in models_to_try:
                try:
                    model = self._genai.GenerativeModel(model_name=model_name)
                    audio_part = {
                        "mime_type": mime_type,
                        "data": audio_bytes
                    }
                    response = model.generate_content([audio_part, prompt])
                    if response and response.text:
                        return response.text.strip().strip('"'), None
                except Exception as e:
                    last_err = str(e)
                    continue

        return None, "Unable to transcribe audio with Gemini API."

    def generate_explanation(self, user_command: str, intent_summary: str,
                             rows_before: int, rows_after: int,
                             sample_df: pd.DataFrame) -> str:
        """Generates a short grounded explanation of the transformation result."""
        if not self._client and not self._genai:
            self._init_client()

        sample_str = sample_df.head(3).to_string(index=False)
        prompt = build_explanation_prompt(user_command, intent_summary, rows_before, rows_after, sample_str)

        if self.api_key:
            try:
                if self._client:
                    res = self._client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    if res and res.text:
                        return res.text.strip()
                elif self._genai:
                    model = self._genai.GenerativeModel(model_name="gemini-2.5-flash")
                    res = model.generate_content(prompt)
                    if res and res.text:
                        return res.text.strip()
            except Exception:
                pass

        # Fallback explanation if API unavailable
        return f"{intent_summary} (Dataset adjusted from {rows_before:,} to {rows_after:,} rows)."
