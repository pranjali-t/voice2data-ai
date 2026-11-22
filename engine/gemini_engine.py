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


def resolve_audio_mime_type(raw_mime: Optional[str] = None, data: bytes = b"") -> str:
    """
    Deterministically detects and standardizes audio MIME types using magic byte inspection
    and MIME string normalization. Ensures compatibility with Google Gemini API.
    """
    # 1. Magic byte inspection for true container format
    if data and len(data) >= 3:
        if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            return "audio/wav"
        if len(data) >= 4 and data.startswith(b"\x1aE\xdf\xa3"):
            return "audio/webm"
        if len(data) >= 4 and data.startswith(b"OggS"):
            return "audio/ogg"
        if len(data) >= 4 and data.startswith(b"fLaC"):
            return "audio/flac"
        if len(data) >= 8 and data[4:8] == b"ftyp":
            return "audio/mp4"
        if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            return "audio/mp3"

    # 2. Sanitize and clean provided raw_mime
    if raw_mime:
        clean = raw_mime.split(";")[0].strip().lower()
        mapping = {
            "audio/wav": "audio/wav",
            "audio/x-wav": "audio/wav",
            "audio/wave": "audio/wav",
            "audio/webm": "audio/webm",
            "audio/ogg": "audio/ogg",
            "audio/mp3": "audio/mp3",
            "audio/mpeg": "audio/mp3",
            "audio/mp4": "audio/mp4",
            "audio/m4a": "audio/mp4",
            "audio/x-m4a": "audio/mp4",
            "audio/aac": "audio/aac",
            "audio/flac": "audio/flac",
            "audio/x-flac": "audio/flac",
        }
        if clean in mapping:
            return mapping[clean]

    return "audio/wav"


def sanitize_error_message(err_msg: Any, api_key: Optional[str] = None) -> str:
    """
    Sanitizes error messages so actual errors are visible for debugging
    while strictly ensuring API keys and credentials are never exposed.
    """
    if not err_msg:
        return "Unknown error"
    text = str(err_msg)
    if api_key and len(api_key) > 5 and api_key in text:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    # Redact common credential patterns (AIza..., key=..., Bearer ...)
    text = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_KEY]", text)
    text = re.sub(r"key=[^&\s\"']+", "key=[REDACTED]", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", text)
    return text.strip()


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
        self._cached_models: Optional[List[str]] = None
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

    def get_supported_models(self) -> List[str]:
        """
        Dynamically queries the Gemini Models API to discover available models that support
        the generateContent method for this API key, prioritizing gemini-3.8-flash first.
        Caches the result to avoid redundant API calls and strictly excludes retired model IDs
        (e.g. gemini-1.5, gemini-2.0, gemini-2.5).
        """
        if self._cached_models:
            return self._cached_models

        # Preferred models ordered by priority: gemini-3.8-flash is candidate #1
        preferred_candidates = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        ]
        retired_keywords = ["1.5", "2.0", "2.5", "gemini-1.5", "gemini-2.0", "gemini-2.5"]

        discovered_supported: List[str] = []

        if self._client:
            try:
                for m in self._client.models.list():
                    m_name = getattr(m, "name", "") or str(m)
                    clean_name = m_name.replace("models/", "").strip()
                    actions = (
                        getattr(m, "supported_actions", None)
                        or getattr(m, "supported_generation_methods", None)
                        or []
                    )
                    if not actions or "generateContent" in actions:
                        if not any(r in clean_name for r in retired_keywords):
                            discovered_supported.append(clean_name)
            except Exception:
                pass

        elif self._genai:
            try:
                for m in self._genai.list_models():
                    m_name = getattr(m, "name", "") or str(m)
                    clean_name = m_name.replace("models/", "").strip()
                    methods = getattr(m, "supported_generation_methods", []) or []
                    if not methods or "generateContent" in methods:
                        if not any(r in clean_name for r in retired_keywords):
                            discovered_supported.append(clean_name)
            except Exception:
                pass

        # Build prioritized list: preferred candidates first if verified in discovered models
        models: List[str] = []
        if discovered_supported:
            for p in preferred_candidates:
                if p in discovered_supported and p not in models:
                    models.append(p)
            for d in discovered_supported:
                if "flash" in d.lower() and d not in models:
                    models.append(d)

        # Fallback if discovery returned empty or ran offline
        if not models:
            models = preferred_candidates.copy()

        self._cached_models = models
        return self._cached_models

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

        models_to_try = self.get_supported_models()
        model_errors: List[str] = []

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
                except Exception as e:
                    err_clean = sanitize_error_message(e, self.api_key)
                    model_errors.append(f"{model_name}: {err_clean}")
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
                except Exception as e:
                    err_clean = sanitize_error_message(e, self.api_key)
                    model_errors.append(f"{model_name}: {err_clean}")
                    continue

        # If all API calls fail, fallback gracefully while recording model attempt details
        fallback_intent = rule_based_fallback_intent(user_command, df)
        err_detail = " | ".join(model_errors) if model_errors else "Gemini API unavailable"
        return fallback_intent, f"Gemini API fallback utilized ({err_detail})"

    def transcribe_audio(self, audio_bytes: bytes, mime_type: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Transcribes speech audio into text using Gemini's native multimodal capabilities.
        
        Args:
            audio_bytes: Raw bytes of the recorded audio.
            mime_type: Optional raw MIME type from the recording source (e.g. 'audio/wav', 'audio/webm').
        
        Returns:
            Tuple of (transcribed_text, error_message)
        """
        if not audio_bytes:
            return None, "No audio data received."

        if not self._client and not self._genai:
            self._init_client()

        if not self.api_key:
            return None, "Gemini API key required for voice transcription. Please configure GEMINI_API_KEY."

        # Detect and normalize true MIME type for audio container
        resolved_mime = resolve_audio_mime_type(raw_mime=mime_type, data=audio_bytes)

        models_to_try = self.get_supported_models()
        prompt = (
            "Transcribe the spoken audio command exactly as spoken. "
            "Return ONLY the plain transcribed words. "
            "Do not include quotes, markdown, extra commentary, formatting, or prefixes. "
            "If the audio contains no intelligible speech, return an empty string."
        )

        model_errors: List[str] = []

        # 1. Modern google.genai SDK
        if self._client:
            from google.genai import types
            for model_name in models_to_try:
                try:
                    part = types.Part.from_bytes(data=audio_bytes, mime_type=resolved_mime)
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=[part, prompt]
                    )
                    transcription = None
                    if response and getattr(response, "text", None):
                        transcription = response.text.strip().strip('"\'')
                    elif response and getattr(response, "candidates", None):
                        for cand in response.candidates:
                            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                                for p in cand.content.parts:
                                    if getattr(p, "text", None):
                                        transcription = p.text.strip().strip('"\'')
                                        break
                    if transcription:
                        cleaned = re.sub(r'^(?:transcription|transcript|transcribed text):\s*', '', transcription, flags=re.IGNORECASE).strip()
                        return cleaned, None
                    elif response:
                        return None, "Audio was recorded, but no spoken speech was detected. Please speak clearly into your microphone."
                except Exception as e:
                    err_clean = sanitize_error_message(e, self.api_key)
                    model_errors.append(f"{model_name}: {err_clean}")
                    continue

        # 2. Fallback to legacy google-generativeai SDK
        if self._genai:
            for model_name in models_to_try:
                try:
                    model = self._genai.GenerativeModel(model_name=model_name)
                    audio_part = {
                        "mime_type": resolved_mime,
                        "data": audio_bytes
                    }
                    response = model.generate_content([audio_part, prompt])
                    transcription = None
                    try:
                        if response and response.text:
                            transcription = response.text.strip().strip('"\'')
                    except Exception:
                        if response and getattr(response, "candidates", None):
                            for cand in response.candidates:
                                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                                    for p in cand.content.parts:
                                        if getattr(p, "text", None):
                                            transcription = p.text.strip().strip('"\'')
                                            break
                    if transcription:
                        cleaned = re.sub(r'^(?:transcription|transcript|transcribed text):\s*', '', transcription, flags=re.IGNORECASE).strip()
                        return cleaned, None
                    elif response:
                        return None, "Audio was recorded, but no spoken speech was detected. Please speak clearly into your microphone."
                except Exception as e:
                    err_clean = sanitize_error_message(e, self.api_key)
                    model_errors.append(f"{model_name}: {err_clean}")
                    continue

        summary_err = " | ".join(model_errors) if model_errors else "API service unavailable"
        return None, f"Transcription failed: {summary_err}"

    def generate_explanation(self, user_command: str, intent_summary: str,
                             rows_before: int, rows_after: int,
                             sample_df: pd.DataFrame) -> str:
        """Generates a short grounded explanation of the transformation result."""
        if not self._client and not self._genai:
            self._init_client()

        sample_str = sample_df.head(3).to_string(index=False)
        prompt = build_explanation_prompt(user_command, intent_summary, rows_before, rows_after, sample_str)

        if self.api_key:
            for model_name in self.get_supported_models():
                try:
                    if self._client:
                        res = self._client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        if res and res.text:
                            return res.text.strip()
                    elif self._genai:
                        model = self._genai.GenerativeModel(model_name=model_name)
                        res = model.generate_content(prompt)
                        if res and res.text:
                            return res.text.strip()
                except Exception:
                    continue

        # Fallback explanation if API unavailable
        return f"{intent_summary} (Dataset adjusted from {rows_before:,} to {rows_after:,} rows)."
