"""Shared styling and small UI helpers."""
import pandas as pd
import streamlit as st

CSS = """
<style>
:root { --qg-navy:#0f2742; --qg-teal:#0e9f8e; --qg-amber:#d98e04; --qg-red:#c0392b; --qg-green:#1e8e3e; }
.qg-hero { background: linear-gradient(120deg, var(--qg-navy), #1d4e7a); color: #fff;
           padding: 28px 32px; border-radius: 14px; margin-bottom: 24px; }
.qg-hero h1 { margin: 0; font-size: 2.4rem; letter-spacing: .5px; color:#fff; }
.qg-hero p { margin: 6px 0 0; opacity: .85; }
.qg-brand { font-weight: 800; font-size: 1.6rem; color: var(--qg-teal); letter-spacing: .5px; }
.qg-badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700;
            font-size: .78rem; color: #fff; }
.qg-verdict { font-size: 2.2rem; font-weight: 800; padding: 10px 22px; border-radius: 12px;
              color: #fff; display: inline-block; letter-spacing: 1px; }
</style>
"""

COLORS = {"PASS": "#1e8e3e", "FAIL": "#c0392b", "REVIEW": "#d98e04", "loaded": "#1e8e3e",
          "unavailable": "#d98e04", "error": "#c0392b", "role": "#1d4e7a", "valid": "#1e8e3e",
          "invalid": "#c0392b"}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_badge(text: str, key: str) -> str:
    return f"<span class='qg-badge' style='background:{COLORS.get(key, '#555')}'>{text}</span>"


def verdict(decision: str) -> str:
    return f"<div class='qg-verdict' style='background:{COLORS[decision]}'>{decision}</div>"


def defects_table(defects: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "#": i + 1, "Class": d["class_name"], "Confidence": round(d["confidence"], 3),
        "x1": d["bbox"]["x1"], "y1": d["bbox"]["y1"], "x2": d["bbox"]["x2"], "y2": d["bbox"]["y2"],
    } for i, d in enumerate(defects)])
