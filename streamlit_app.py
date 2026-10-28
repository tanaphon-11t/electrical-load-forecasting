"""
streamlit_app_merged.py
Merged UI from streamlit_app(4).py + stable app flow from streamlit_app1.py.
Fixed Data Update editor reset / empty-column issue.

Original base: streamlit_app.py
Final Fixed App - Electricity Demand Forecasting

Flow:
1) Upload Excel
2) Update Dataset
3) Train / Forecast
4) Predict Next Month
5) Accuracy Check
ุ6) Export Excel
"""
PROVINCE_COORDS = {
    "Bangkok": (13.7563, 100.5018),
    "Ang Thong": (14.5896, 100.4550),
    "Chachoengsao": (13.6904, 101.0779),
    "Chanthaburi": (12.6113, 102.1038),
    "Chonburi": (13.3611, 100.9847),
    "Kanchanaburi": (14.0228, 99.5328),
    "Nakhon Nayok": (14.2069, 101.2131),
    "Nakhon Pathom": (13.8199, 100.0622),
    "Nonthaburi": (13.8591, 100.5217),
    "Pathum Thani": (14.0208, 100.5250),
    "Phra Nakhon Si Ayutthaya": (14.3532, 100.5689),
    "Prachin Buri": (14.0500, 101.3700),
    "Rayong": (12.6814, 101.2816),
    "Sa Kaeo": (13.8240, 102.0646),
    "Samut Prakan": (13.5991, 100.5998),
    "Samut Sakhon": (13.5475, 100.2744),
    "Saraburi": (14.5289, 100.9101),
    "Suphan Buri": (14.4745, 100.1177),
    "Trat": (12.2428, 102.5175),
}

from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import config as st_config

from forecasting_engine import (
    MODEL_VERSION,
    forecast_all_provinces,
    forecast_province,
    create_summary_dataframe,
    predict_next_month,
)
from ai_methods_v2.registry import METHOD_REGISTRY
from benchmark import metric_summary

APP_DIR = Path(__file__).resolve().parent
APP_FILE = Path(__file__).name
ENGINE_FILE = "forecasting_engine.py"
EXPERIMENT_FILE = APP_DIR / "ผลทดลอง" / "ชุดข้อมูลและผลทดลองปัจจัยปฏิทิน_19จังหวัด_มค-มิย2026.xlsx"
EXPECTED_PROVINCE_COUNT = 19
DATA_SOURCE_NAME = "Energy e-Report, Ministry of Energy"
TARGET_DISPLAY_NAME = "Provincial Load Proxy"
TARGET_DEFINITION_TH = (
    "ปริมาณการใช้ไฟฟ้ารายเดือนของหน่วยงานภาครัฐในแต่ละจังหวัด "
    "ที่รายงานผ่านระบบ Energy e-Report"
)
OUTLIER_THRESHOLD_PCT = 30.0
EXTERNAL_CONTEXT_SOURCES = "NASA POWER / PEA / MEA / EGAT"
SHOW_ADVANCED_TOOLS = False
SHOW_APP_BANNER = True
SHOW_TARGET_DEFINITION_NOTICE = False
AI_TRAINING_MAX_WORKERS = 4
AI_METHOD_LABELS = {
    "chatgpt_carr1m": "ChatGPT — CARR-1M",
    "gemini_cg_hae_v1": "Gemini — CG-HAE-v1",
    "kimi_pgf_tea": "Kimi — PGF-TEA",
}

# =========================
# PAGE CONFIG
# =========================
# Enforce the dark theme from the app itself. This keeps DataFrame/DataEditor
# dark even when the user starts Streamlit from a directory that does not load
# app4/.streamlit/config.toml.
for option_name, option_value in {
    "theme.base": "dark",
    "theme.primaryColor": "#00E5FF",
    "theme.backgroundColor": "#260007",
    "theme.secondaryBackgroundColor": "#111C31",
    "theme.textColor": "#F8FAFC",
}.items():
    try:
        st_config.set_option(option_name, option_value)
    except Exception:
        pass

st.set_page_config(
    page_title="ระบบพยากรณ์ความต้องการใช้ไฟฟ้า",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg0: #260007;
    --bg1: #5a0b18;
    --surface: rgba(15, 23, 42, 0.92);
    --surface-soft: rgba(30, 41, 59, 0.88);
    --border: rgba(56, 189, 248, 0.26);
    --text: #f8fafc;
    --text-strong: #ffffff;
    --muted: #cbd5e1;
    --primary: #00e5ff;
    --primary-2: #2563eb;
    --accent: #a78bfa;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], button, input, textarea, select,
div, span, p, label {
    font-family: 'Kanit', sans-serif !important;
}

.stApp {
    background-color: var(--bg1);
    background-image:
        radial-gradient(circle at top left, rgba(220,24,58,0.38), transparent 32rem),
        radial-gradient(circle at top right, rgba(158,10,36,0.42), transparent 30rem),
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(180deg, #260007 0%, #5a0b18 52%, #260007 100%);
    background-size: auto, auto, 40px 40px, 40px 40px, auto;
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1380px;
}

[data-testid="stHeader"] {
    background: rgba(7,17,31,0.78);
    backdrop-filter: blur(14px);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-strong) !important;
    font-weight: 800 !important;
}

p, li, label, span, div, .stCaptionContainer, [data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

/* Hero */
.hero {
    overflow: hidden;
    border: 1px solid rgba(0,229,255,0.22);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(17,24,39,0.88));
    box-shadow: 0 0 24px rgba(0,229,255,0.18), 0 18px 52px rgba(0,0,0,0.38);
    margin-bottom: 1rem;
}
.hero-image img, [data-testid="stImageContainer"] img {
    display: block;
    width: 100%;
    max-height: 160px;
    object-fit: cover;
}
.hero-copy {
    padding: 1.15rem 1.5rem 1.25rem;
    border-top: 1px solid rgba(56,189,248,0.20);
}
.eyebrow {
    color: var(--primary) !important;
    font-size: .84rem;
    font-weight: 800;
    letter-spacing: .09em;
    margin-bottom: .35rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: clamp(1.9rem, 3.5vw, 3rem);
    font-weight: 900 !important;
    line-height: 1.08;
    margin: 0 0 .65rem;
    background: linear-gradient(90deg, #00e5ff, #a78bfa, #ff4ecd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 26px rgba(0,229,255,0.18);
}
.hero-subtitle { color: var(--muted) !important; font-size: 1rem; line-height: 1.7; margin: 0; }

.project-meta { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:.75rem; margin:0 0 1rem; }
.project-meta-card, .status-item, [data-testid="metric-container"], .app-footer {
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.78));
    box-shadow: 0 0 18px rgba(0,229,255,0.12), 0 10px 26px rgba(0,0,0,0.22);
}
.project-meta-card { display:flex; align-items:flex-start; gap:.7rem; min-height:86px; padding:1rem; }
.project-meta-icon { display:grid; place-items:center; flex:0 0 2.35rem; width:2.35rem; height:2.35rem; border-radius:14px; background:rgba(0,229,255,.12); color:var(--primary) !important; font-size:1.15rem; }
.project-meta-label, .status-label, .footer-label { color: var(--muted) !important; font-size:.82rem; font-weight:700; }
.project-meta-value, .status-value, .footer-value { color: var(--text-strong) !important; font-weight:800; }

.section-header {
    display:flex; align-items:center; justify-content:space-between; gap:1rem;
    margin:1rem 0; padding:1rem 1.15rem;
    border: 1px solid var(--border);
    border-left: 5px solid var(--primary);
    border-radius:18px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.80));
    box-shadow: 0 0 18px rgba(0,229,255,0.14);
}
.section-title { color:var(--text-strong) !important; font-size:1.28rem; font-weight:900; line-height:1.25; margin:0; }
.section-caption { color:var(--muted) !important; font-size:.94rem; margin-top:.2rem; }
.status-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin:.5rem 0 1.2rem; }
.status-item { padding:.85rem 1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:.45rem; border:1px solid var(--border); border-radius:18px; padding:.45rem; background:rgba(15,23,42,.88); box-shadow:0 0 16px rgba(0,229,255,.10); }
.stTabs [data-baseweb="tab"] { height:52px; border-radius:14px; background:rgba(30,41,59,.68); color:var(--muted) !important; font-size:.98rem; font-weight:800; padding:0 1rem; border:1px solid rgba(148,163,184,.16); }
.stTabs [data-baseweb="tab"]:hover { background:linear-gradient(135deg, rgba(37,99,235,.55), rgba(6,182,212,.42)); color:white !important; box-shadow:0 0 18px rgba(0,229,255,.24); }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg, #00b7ff, #2563eb) !important; color:#fff !important; border-color:rgba(255,255,255,.30); box-shadow:0 0 22px rgba(0,229,255,.45); }
.stTabs [data-baseweb="tab-highlight"] { display:none; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius:14px; border:1px solid rgba(56,189,248,.35);
    background:linear-gradient(135deg, rgba(15,23,42,.95), rgba(30,41,59,.88));
    color:var(--text-strong) !important; font-weight:800; min-height:2.75rem;
    box-shadow:0 0 12px rgba(0,229,255,.14);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--primary); color:#fff !important; box-shadow:0 0 22px rgba(0,229,255,.30); transform:translateY(-1px); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { border-color:rgba(255,255,255,.22); background:linear-gradient(135deg,#00e5ff,#2563eb) !important; color:white !important; }

/* Inputs */
[data-baseweb="input"] input, [data-baseweb="select"] div, textarea, input {
    color: #ffffff !important;
}
[data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
    background: rgba(15,23,42,.95) !important;
    border-color: rgba(56,189,248,.28) !important;
    color: #ffffff !important;
}

/* Streamlit renders select/file controls with an internal light theme.
   Override the real inner containers so white text never sits on white. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background: #111c31 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56,189,248,.48) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stMultiSelect"] [data-baseweb="select"] span,
[data-baseweb="popover"] [role="option"] {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}
[data-testid="stSelectbox"] svg,
[data-testid="stMultiSelect"] svg {
    fill: #67e8f9 !important;
    color: #67e8f9 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: #1d4ed8 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #111c31, #17233a) !important;
    border: 1px dashed rgba(56,189,248,.62) !important;
    box-shadow: inset 0 0 18px rgba(0,229,255,.06);
}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.24) !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: #67e8f9 !important;
    color: #67e8f9 !important;
}

/* Native text/date/number fields */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background: #111c31 !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}

/* DataFrame / editor contrast + keep Streamlit toolbar/fullscreen usable */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius:16px;
    overflow:visible !important;
    border:1px solid rgba(56,189,248,.22);
}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * { color: #f8fafc !important; }
[data-testid="stDataFrame"] div[role="columnheader"], [data-testid="stDataEditor"] div[role="columnheader"] { color:#dbeafe !important; font-weight:800 !important; }
[data-testid="stDataFrame"] [data-testid="stElementToolbar"],
[data-testid="stDataEditor"] [data-testid="stElementToolbar"],
[data-testid="stElementToolbar"] {
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 9999 !important;
    pointer-events: auto !important;
}
[data-testid="stDataFrame"] button,
[data-testid="stDataEditor"] button,
[data-testid="stElementToolbar"] button {
    pointer-events: auto !important;
    color: #ffffff !important;
    background: rgba(15,23,42,.88) !important;
    border: 1px solid rgba(56,189,248,.35) !important;
}

.stAlert { border-radius:16px; }
.stAlert * { color: inherit !important; }

.app-footer { margin-top:1.5rem; padding:1.15rem 1.25rem; }
.footer-title { color:var(--text-strong) !important; font-size:1.05rem; font-weight:900; margin-bottom:.2rem; }
.footer-grid { display:grid; grid-template-columns:1.2fr 1fr 1fr; gap:1rem; margin-top:.9rem; }

@media (max-width: 900px) {
    .status-strip { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .project-meta, .footer-grid { grid-template-columns:1fr; }
    .section-header { align-items:flex-start; flex-direction:column; }
}

/* Final-project control-room skin. Presentation only; widget keys and state stay unchanged. */
.block-container { max-width: 1540px; padding-top: .7rem; }
.hero { border-radius: 12px; margin-bottom: .65rem; border-color: rgba(239,68,68,.35); }
.hero-image img, [data-testid="stImageContainer"] img {
    width:100% !important;
    height:auto !important;
    max-height:none !important;
    object-fit:contain !important;
}
.hero-copy { display:none; }
.project-meta { display:none; }
.section-header {
    border-radius: 10px; border-left: 0; margin:.3rem 0 .55rem; padding:.62rem .9rem;
    background:linear-gradient(115deg,rgba(45,10,16,.96),rgba(13,20,32,.94));
    border-color:rgba(239,68,68,.26); box-shadow:0 10px 30px rgba(0,0,0,.28);
}
.section-title { font-size:1.22rem; line-height:1.15; }
.section-caption { font-size:.82rem; line-height:1.35; margin-top:.12rem; }
.stTabs [data-baseweb="tab-list"] { border-radius:12px; background:rgba(29,8,13,.94); border-color:rgba(239,68,68,.25); }
.stTabs [data-baseweb="tab"] { border-radius:8px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1d4ed8,#254cff) !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:rgba(148,163,184,.20) !important; border-radius:12px !important;
}
.ai-console-title { display:flex; align-items:center; gap:.75rem; margin:.25rem 0 .1rem; }
.ai-console-icon { font-size:2rem; }
.ai-console-title strong { font-size:1.45rem; color:#f8fafc; }
.ai-console-subtitle { color:#cbd5e1 !important; margin-left:3rem; margin-bottom:.75rem; }
.method-card {
    min-height:318px; padding:1rem; border-radius:14px; margin:.1rem 0 .7rem;
    background:linear-gradient(145deg,rgba(18,28,32,.97),rgba(18,24,34,.91));
    border:1px solid var(--method-color); box-shadow:0 12px 28px rgba(0,0,0,.25);
}
.method-card.chatgpt { --method-color:#20b981; }
.method-card.gemini { --method-color:#2997ff; background:linear-gradient(145deg,rgba(12,25,39,.97),rgba(18,24,34,.91)); }
.method-card.kimi { --method-color:#f68712; background:linear-gradient(145deg,rgba(45,25,14,.96),rgba(28,20,22,.92)); }
.method-brand { color:var(--method-color) !important; font-size:1.2rem; font-weight:800; margin-bottom:.15rem; }
.method-name { color:#e2e8f0 !important; font-size:.9rem; margin-bottom:.85rem; }
.method-status { display:inline-block; padding:.25rem .55rem; border-radius:6px; background:rgba(30,41,59,.85); color:#fff !important; font-size:.78rem; font-weight:700; }
.method-detail { margin-top:.85rem; padding:.8rem; border:1px solid rgba(148,163,184,.18); border-radius:9px; background:rgba(255,255,255,.035); color:#dbe4ee !important; font-size:.86rem; line-height:1.65; }
.method-detail b { color:#fff !important; }
.training-empty { text-align:center; padding:2.4rem 1rem; border:1px solid rgba(148,163,184,.18); border-radius:12px; background:rgba(15,23,42,.55); color:#cbd5e1 !important; }
.main-system-title {
    margin:.45rem 0 .55rem;
    padding:.55rem .9rem;
    border-left:4px solid #f59e0b;
    border-radius:7px;
    background:linear-gradient(90deg,rgba(35,4,8,.96),rgba(45,8,13,.72),transparent);
    color:#fbbf24 !important;
    font-size:1.35rem;
    font-weight:900;
    line-height:1.45;
    white-space:nowrap;
}

/* Desktop application shell: the existing six functional tabs become a fixed sidebar. */
@media (min-width: 701px) {
    .block-container,
    [data-testid="stAppViewContainer"] .block-container {
        width:calc(100% - 282px) !important;
        max-width:none !important;
        margin-left:282px !important;
        padding-left:1rem !important;
        padding-right:1rem !important;
    }
    header[data-testid="stHeader"] { height:0 !important; min-height:0 !important; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
    .stTabs [data-baseweb="tab-list"] {
        position:fixed !important;
        z-index:999;
        left:6px;
        top:6px;
        bottom:6px;
        width:264px;
        height:auto;
        box-sizing:border-box;
        display:flex !important;
        flex-direction:column !important;
        align-items:stretch !important;
        justify-content:flex-start !important;
        gap:.38rem !important;
        padding:18px 12px 196px !important;
        border:1px solid rgba(239,68,68,.28) !important;
        border-radius:10px !important;
        background:
            radial-gradient(circle at 20% 35%,rgba(120,16,32,.26),transparent 55%),
            linear-gradient(180deg,#180306 0%,#26070d 48%,#160306 100%) !important;
        box-shadow:10px 0 35px rgba(0,0,0,.36) !important;
        overflow:hidden !important;
    }
    .stTabs [data-baseweb="tab-list"]::before {
        display:none !important;
    }
    .stTabs [data-baseweb="tab-list"]::after {
        content:"ข้อมูลระบบ\\A ───────────────\\A ▪ เวอร์ชันแอป 2.0.0\\A ▪ Production Dataset\\A ▪ AI Methods 3 ระบบ";
        white-space:pre-line;
        position:absolute;
        left:16px;
        right:16px;
        bottom:26px;
        padding-top:14px;
        border-top:1px solid rgba(148,163,184,.22);
        color:#cbd5e1 !important;
        font-size:.78rem;
        font-weight:500;
        line-height:1.8;
    }
    .stTabs [data-baseweb="tab"] {
        width:100% !important;
        height:46px !important;
        flex:none !important;
        justify-content:flex-start !important;
        padding:0 .8rem !important;
        border:1px solid transparent !important;
        background:transparent !important;
        border-radius:7px !important;
        color:#e2e8f0 !important;
        box-shadow:none !important;
    }
    .stTabs [data-baseweb="tab"] p { font-size:.94rem !important; font-weight:500 !important; }
    .stTabs [data-baseweb="tab"]:hover {
        background:rgba(37,99,235,.28) !important;
        border-color:rgba(96,165,250,.25) !important;
    }
    .stTabs [aria-selected="true"] {
        background:linear-gradient(135deg,#2452ff,#1d4ed8) !important;
        border-color:rgba(96,165,250,.55) !important;
        box-shadow:0 6px 16px rgba(37,99,235,.32) !important;
    }
    .stTabs [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"] { display:none !important; }
    div[data-testid="stHorizontalBlock"]:has(+ div[data-testid="stTabs"]) {
        position:fixed !important;
        z-index:1001 !important;
        left:18px !important;
        bottom:136px !important;
        width:238px !important;
        height:42px !important;
        min-height:42px !important;
        margin:0 !important;
        gap:0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(+ div[data-testid="stTabs"]) > div:first-child {
        display:none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(+ div[data-testid="stTabs"]) > div:nth-child(2) {
        display:block !important;
        width:100% !important;
        flex:1 1 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(+ div[data-testid="stTabs"]) button {
        width:100% !important;
    }
}

/* Shared panels used by Data Import, Forecasting, Accuracy, Update and Reports. */
[data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stDataFrame"],
[data-testid="stDataEditor"], [data-testid="stFileUploader"],
[data-testid="stDownloadButton"], [data-testid="stForm"] {
    border-radius:10px !important;
}
[data-testid="stMetric"] {
    background:linear-gradient(145deg,rgba(25,24,29,.94),rgba(28,31,39,.90));
    border:1px solid rgba(148,163,184,.18);
    padding:.75rem 1rem;
}
[data-testid="stAlert"] {
    background:linear-gradient(135deg,rgba(43,35,46,.91),rgba(39,40,49,.88)) !important;
    border:1px solid rgba(148,163,184,.16) !important;
    padding:.55rem .75rem !important;
}
.stButton > button, .stDownloadButton > button {
    border-radius:7px !important;
    min-height:2.55rem;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    border-radius:7px !important;
    background:#242832 !important;
    border-color:rgba(148,163,184,.22) !important;
}
.app-footer { background:linear-gradient(135deg,rgba(25,20,26,.96),rgba(22,26,35,.92)); border-color:rgba(239,68,68,.18); }

@media (max-width: 700px) {
    .block-container { width:100% !important; margin-left:0 !important; }
    .stTabs [data-baseweb="tab-list"] {
        position:static !important;
        display:flex !important;
        flex-direction:row !important;
        overflow-x:auto !important;
    }
    .method-card { min-height:auto; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div[data-testid="stElementContainer"]:has(.legacy-ui-hide-from-here),
    div[data-testid="stElementContainer"]:has(.legacy-ui-hide-from-here)
      ~ div[data-testid="stElementContainer"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# SESSION STATE INIT
# ==========================
if "df" not in st.session_state:
    st.session_state.df = None

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "forecasted" not in st.session_state:
    st.session_state.forecasted = False

if "all_results" not in st.session_state:
    st.session_state.all_results = None

if "summary_df" not in st.session_state:
    st.session_state.summary_df = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()

if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = pd.DataFrame()

if "external_validation_df" not in st.session_state:
    st.session_state.external_validation_df = pd.DataFrame()

# =========================
# HELPERS
# =========================
def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean uploaded Excel data."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Province", "Date", "Load_(kWh)", "Temperature_C"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์: {missing}")

    df["Province"] = df["Province"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Load_(kWh)"] = (
        df["Load_(kWh)"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
    )

    df["Load_(kWh)"] = pd.to_numeric(df["Load_(kWh)"], errors="coerce")
    

    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors="coerce")

    if "Month_Num" not in df.columns:
        df["Month_Num"] = df["Date"].dt.month
    else:
        df["Month_Num"] = pd.to_numeric(df["Month_Num"], errors="coerce")
        df["Month_Num"] = df["Month_Num"].fillna(df["Date"].dt.month)

    df = df.dropna(
       subset=["Province", "Date", "Load_(kWh)", "Temperature_C"]
    ).copy()

    df = df.sort_values(["Province", "Date"]).reset_index(drop=True)
    return df


def get_date_status(df: pd.DataFrame):
    today = pd.Timestamp.today()
    current_month = today.to_period("M").to_timestamp()

    latest_date = pd.to_datetime(df["Date"]).max()
    latest_month = latest_date.to_period("M").to_timestamp()
    next_month = latest_month + pd.DateOffset(months=1)

    status = (
        f"🟢 ข้อมูลครบถึง {latest_month.strftime('%Y-%m')} "
        f"สามารถพยากรณ์เดือน {next_month.strftime('%Y-%m')} ได้"
    )

    return current_month, latest_month, next_month, status

def get_nasa_monthly_temp(lat, lon, year, month):
    import requests

    start = f"{year}{month:02d}01"
    end = f"{year}{month:02d}28"

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        f"&format=JSON"
    )

    response = requests.get(url, timeout=8)
    response.raise_for_status()

    data = response.json()
    temp_dict = data["properties"]["parameter"]["T2M"]

    temps = [v for v in temp_dict.values() if v is not None and v > -99]

    if not temps:
        raise ValueError("NASA ไม่มีข้อมูลอุณหภูมิ")

    return sum(temps) / len(temps)


def get_automatic_temperature(
    df: pd.DataFrame,
    province: str,
    forecast_month: pd.Timestamp,
):
    """Use NASA when available; otherwise use a reproducible historical fallback."""
    lat, lon = PROVINCE_COORDS[province]
    try:
        value = get_nasa_monthly_temp(
            lat,
            lon,
            forecast_month.year,
            forecast_month.month,
        )
        return round(float(value), 2), "NASA POWER"
    except Exception:
        province_df = df[df["Province"] == province].copy()
        same_month = province_df[
            province_df["Date"].dt.month == forecast_month.month
        ]["Temperature_C"].dropna()
        if not same_month.empty:
            return round(float(same_month.median()), 2), "Historical monthly median fallback"
        return (
            round(float(province_df.sort_values("Date")["Temperature_C"].iloc[-1]), 2),
            "Latest temperature fallback",
        )


def find_trained_result(all_results, province: str):
    if isinstance(all_results, list):
        return next(
            (result for result in all_results if result.get("Province") == province),
            None,
        )
    if isinstance(all_results, dict):
        return all_results.get(province)
    return None


def get_all_automatic_temperatures(
    df: pd.DataFrame,
    provinces,
    forecast_month: pd.Timestamp,
):
    """Fetch independent province temperatures in parallel for one-click runs."""
    results = {}
    with ThreadPoolExecutor(max_workers=min(8, max(len(provinces), 1))) as pool:
        futures = {
            pool.submit(
                get_automatic_temperature,
                df,
                province,
                forecast_month,
            ): province
            for province in provinces
        }
        for future in as_completed(futures):
            province = futures[future]
            try:
                results[province] = future.result()
            except Exception:
                province_df = df[df["Province"] == province].sort_values("Date")
                results[province] = (
                    round(float(province_df["Temperature_C"].iloc[-1]), 2),
                    "Latest temperature fallback",
                )
    return results


def latest_status_table(df: pd.DataFrame) -> pd.DataFrame:
    latest_by_province = (
        df.groupby("Province", as_index=False)["Date"]
        .max()
        .rename(columns={"Date": "Latest_Date"})
    )
    overall_latest = latest_by_province["Latest_Date"].max()
    latest_by_province["Latest_Month"] = latest_by_province["Latest_Date"].dt.strftime("%Y-%m")
    latest_by_province["Months_Behind"] = latest_by_province["Latest_Date"].apply(
        lambda x: (overall_latest.year - x.year) * 12 + (overall_latest.month - x.month)
    )
    latest_by_province["Status"] = latest_by_province["Months_Behind"].apply(
        lambda m: "✅ ล่าสุด" if m == 0 else "⚠️ ต้องเติมข้อมูล"
    )
    return latest_by_province


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the latest two reported months and flag possible coverage issues."""
    ordered = df.sort_values(["Province", "Date"]).copy()
    ordered["Previous_Load_kWh"] = ordered.groupby("Province")["Load_(kWh)"].shift(1)
    ordered["MoM_Change_%"] = (
        (ordered["Load_(kWh)"] - ordered["Previous_Load_kWh"])
        / ordered["Previous_Load_kWh"]
        * 100
    )
    latest_rows = ordered.groupby("Province", as_index=False).tail(1).copy()
    latest_rows["Data_Quality_Status"] = np.select(
        [
            latest_rows["Previous_Load_kWh"].isna(),
            latest_rows["MoM_Change_%"].abs() > OUTLIER_THRESHOLD_PCT,
        ],
        [
            "⚠️ ไม่มีเดือนก่อนหน้า",
            f"🚨 ตรวจสอบ Coverage (เกิน ±{OUTLIER_THRESHOLD_PCT:.0f}%)",
        ],
        default="✅ อยู่ในเกณฑ์ตรวจสอบปกติ",
    )
    latest_rows["Data_Source"] = DATA_SOURCE_NAME
    latest_rows["Target_Type"] = TARGET_DISPLAY_NAME
    return latest_rows[
        [
            "Province",
            "Date",
            "Previous_Load_kWh",
            "Load_(kWh)",
            "MoM_Change_%",
            "Data_Quality_Status",
            "Data_Source",
            "Target_Type",
        ]
    ].sort_values("MoM_Change_%")


def build_data_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("target_name", TARGET_DISPLAY_NAME),
            ("operational_definition_th", TARGET_DEFINITION_TH),
            ("primary_source", DATA_SOURCE_NAME),
            ("coverage", "หน่วยงานภาครัฐที่ส่งข้อมูลเข้าสู่ e-Report"),
            ("not_equivalent_to", "โหลดไฟฟ้ารวมของบ้าน ธุรกิจ และอุตสาหกรรมทั้งจังหวัด"),
            ("external_validation", "PEA / MEA / EGAT ใช้ตรวจทิศทาง ไม่ปนกับ Training Data"),
            ("outlier_rule", f"แจ้งเตือนเมื่อเดือนต่อเดือนเกิน ±{OUTLIER_THRESHOLD_PCT:.0f}%"),
        ],
        columns=["Field", "Definition"],
    )


def find_metric_col(summary_df: pd.DataFrame, keyword: str):
    matches = [c for c in summary_df.columns if keyword.upper() in str(c).upper()]
    return matches[0] if matches else None


def train_ai_methods(
    df_train: pd.DataFrame,
    provinces: list[str],
    method_ids: list[str],
    train_cutoff: pd.Timestamp,
    progress_callback=None,
):
    """Train frozen AI methods with bounded orchestration concurrency."""
    artifacts = {method_id: {} for method_id in method_ids}
    rows = []

    def train_one(method_id, province):
        method_class = METHOD_REGISTRY[method_id]
        province_history = df_train[
            df_train["Province"].astype(str).str.strip() == str(province).strip()
        ].copy()
        try:
            artifact = method_class().train(
                province_history,
                train_cutoff=train_cutoff,
            )
            metrics = artifact.get("metrics", artifact)
            mape = float(metrics.get("MAPE", metrics.get("mape")))
            rmse = float(metrics.get("RMSE", metrics.get("rmse")))
            r2 = float(metrics.get("R2", metrics.get("r2")))
            if not np.isfinite([mape, rmse, r2]).all():
                raise ValueError("validation metrics contain NaN or infinity")
            artifact_result = artifact
            row = {
                "Method_ID": method_id,
                "Method": AI_METHOD_LABELS[method_id],
                "Province": province,
                "MAPE": mape,
                "RMSE": rmse,
                "R2": r2,
                "Status": "PASS" if mape <= 10.0 else "FAIL",
                "Error": "",
            }
        except Exception as exc:
            artifact_result = None
            row = {
                "Method_ID": method_id,
                "Method": AI_METHOD_LABELS[method_id],
                "Province": province,
                "MAPE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "Status": "ERROR",
                "Error": str(exc),
            }
        return method_id, province, artifact_result, row

    tasks = [(method_id, province) for method_id in method_ids for province in provinces]
    with ThreadPoolExecutor(
        max_workers=min(AI_TRAINING_MAX_WORKERS, max(len(tasks), 1))
    ) as pool:
        futures = {
            pool.submit(train_one, method_id, province): (method_id, province)
            for method_id, province in tasks
        }
        completed = 0
        for future in as_completed(futures):
            method_id, province, artifact, row = future.result()
            if artifact is not None:
                artifacts[method_id][province] = artifact
            rows.append(row)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(tasks), method_id, province)

    method_order = {method_id: index for index, method_id in enumerate(method_ids)}
    province_order = {province: index for index, province in enumerate(provinces)}
    rows.sort(key=lambda row: (method_order[row["Method_ID"]], province_order[row["Province"]]))
    return artifacts, pd.DataFrame(rows)


def summarize_ai_training(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build method-level summaries from frozen internal-validation metrics."""
    rows = []
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    for method_id, group in results_df.groupby("Method_ID", sort=False):
        valid = group.dropna(subset=["MAPE", "RMSE", "R2"])
        rows.append({
            "Method": AI_METHOD_LABELS.get(method_id, method_id),
            "Avg MAPE": valid["MAPE"].mean() if not valid.empty else np.nan,
            "Avg RMSE": valid["RMSE"].mean() if not valid.empty else np.nan,
            "Avg R2": valid["R2"].mean() if not valid.empty else np.nan,
            "PASS": int((valid["MAPE"] <= 10.0).sum()),
            "FAIL": int((valid["MAPE"] > 10.0).sum()),
            "ERROR": int((group["Status"] == "ERROR").sum()),
            "Total": int(len(group)),
        })
    return pd.DataFrame(rows)


def predict_ai_methods(
    df: pd.DataFrame,
    method_results: dict,
    method_ids: list[str],
    provinces: list[str],
    temperatures: dict[str, float],
) -> pd.DataFrame:
    """Predict with already-trained method/province artifacts only."""
    rows = []
    for method_id in method_ids:
        method_class = METHOD_REGISTRY[method_id]
        for province in provinces:
            artifact = method_results.get(method_id, {}).get(province)
            if artifact is None:
                raise ValueError(
                    f"ไม่พบ trained result: {AI_METHOD_LABELS[method_id]} / {province}"
                )
            next_temperature = float(temperatures[province])
            cutoff = pd.Timestamp(artifact["train_cutoff"])
            history = df[
                (df["Province"].astype(str).str.strip() == str(province).strip())
                & (pd.to_datetime(df["Date"]) <= cutoff)
            ].copy()
            prediction = method_class().predict(
                artifact,
                history,
                next_temperature=next_temperature,
            )
            row = prediction.iloc[0].to_dict()
            row["Forecast_Month"] = pd.Timestamp(row["Forecast_Month"]).strftime("%Y-%m")
            row["Run_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows.append(row)
    return pd.DataFrame(rows)


def merge_prediction_history(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep one prediction per method/province/month without cross-method overwrite."""
    existing = existing.copy() if existing is not None else pd.DataFrame()
    new_rows = new_rows.copy()
    for frame in (existing, new_rows):
        if "Method_ID" not in frame.columns:
            frame["Method_ID"] = "legacy"
        else:
            frame["Method_ID"] = frame["Method_ID"].fillna("legacy")
        if "Method_Name" not in frame.columns:
            frame["Method_Name"] = "Legacy Forecasting Engine"
        else:
            frame["Method_Name"] = frame["Method_Name"].fillna("Legacy Forecasting Engine")
    combined = pd.concat([existing, new_rows], ignore_index=True)
    return combined.drop_duplicates(
        subset=["Method_ID", "Province", "Forecast_Month"], keep="last"
    ).reset_index(drop=True)


def calculate_method_accuracy(
    predictions: pd.DataFrame, actual: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare frozen AI predictions with unique raw actual observations."""
    actual = actual.copy()
    actual.columns = [str(column).strip() for column in actual.columns]
    required = ["Province", "Date", "Actual_Load_kWh"]
    missing = [column for column in required if column not in actual.columns]
    if missing:
        raise ValueError(f"ไฟล์ Actual ขาดคอลัมน์: {missing}")
    actual["Province"] = actual["Province"].astype(str).str.strip()
    actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")
    actual["Forecast_Month"] = actual["Date"].dt.strftime("%Y-%m")
    actual["Actual_Load_kWh"] = pd.to_numeric(
        actual["Actual_Load_kWh"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    actual = actual.dropna(
        subset=["Province", "Forecast_Month", "Actual_Load_kWh"]
    ).copy()
    if actual.duplicated(["Province", "Forecast_Month"]).any():
        raise ValueError("Actual ต้องไม่ซ้ำใน Province + Forecast_Month")
    if (actual["Actual_Load_kWh"] == 0).any():
        raise ValueError("Actual_Load_kWh ต้องไม่เป็นศูนย์สำหรับ MAPE")

    predictions = predictions.copy()
    if "Method_ID" not in predictions.columns:
        predictions["Method_ID"] = "legacy"
    if "Method_Name" not in predictions.columns:
        predictions["Method_Name"] = "Legacy Forecasting Engine"
    if predictions.empty:
        raise ValueError("ไม่พบ Prediction ของ AI Methods")
    predictions["Province"] = predictions["Province"].astype(str).str.strip()
    predictions["Forecast_Month"] = predictions["Forecast_Month"].astype(str)
    predictions = predictions.drop_duplicates(
        ["Method_ID", "Province", "Forecast_Month"], keep="last"
    )
    method_keys = predictions[["Method_ID", "Method_Name"]].drop_duplicates()
    expanded_actual = actual.merge(method_keys, how="cross")
    merged = predictions.merge(
        expanded_actual[
            ["Method_ID", "Province", "Forecast_Month", "Actual_Load_kWh"]
        ],
        on=["Method_ID", "Province", "Forecast_Month"],
        how="left",
    )
    merged["Error_%"] = (
        (merged["Actual_Load_kWh"] - merged["Predicted_Load_kWh"]).abs()
        / merged["Actual_Load_kWh"].abs()
        * 100.0
    )
    merged["Accuracy_Status"] = np.where(
        merged["Error_%"].isna(),
        "MISSING ACTUAL",
        np.where(merged["Error_%"] <= 10.0, "PASS", "FAIL"),
    )

    summary_rows = []
    checked = merged.dropna(subset=["Actual_Load_kWh", "Predicted_Load_kWh"])
    ai_checked = checked[checked["Method_ID"].isin(AI_METHOD_LABELS)].copy()
    if ai_checked.empty:
        return merged, pd.DataFrame(), pd.DataFrame()
    for method_id, group in ai_checked.groupby("Method_ID", sort=False):
        metrics = metric_summary(group["Actual_Load_kWh"], group["Predicted_Load_kWh"])
        summary_rows.append({
            "Method_ID": method_id,
            "Method": AI_METHOD_LABELS[method_id],
            "Avg MAPE": metrics["MAPE_pct"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "PASS": int((group["Error_%"] <= 10.0).sum()),
            "FAIL": int((group["Error_%"] > 10.0).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    comparison = actual[["Province", "Forecast_Month", "Actual_Load_kWh"]].merge(
        ai_checked.pivot_table(
            index=["Province", "Forecast_Month"],
            columns="Method_ID",
            values="Error_%",
            aggfunc="last",
        ).reset_index(),
        on=["Province", "Forecast_Month"],
        how="inner",
    )
    comparison = comparison.rename(columns={
        "Actual_Load_kWh": "Actual",
        "chatgpt_carr1m": "ChatGPT Error%",
        "gemini_cg_hae_v1": "Gemini Error%",
        "kimi_pgf_tea": "Kimi Error%",
    })
    error_columns = [
        column for column in ["ChatGPT Error%", "Gemini Error%", "Kimi Error%"]
        if column in comparison.columns
    ]
    label_by_column = {
        "ChatGPT Error%": "ChatGPT — CARR-1M",
        "Gemini Error%": "Gemini — CG-HAE-v1",
        "Kimi Error%": "Kimi — PGF-TEA",
    }
    comparison["Best Method"] = comparison[error_columns].idxmin(axis=1).map(label_by_column)
    return merged, summary, comparison


def call_predict(df: pd.DataFrame, province: str, next_temp: float, alpha: float, trained_result=None):
    """Call forecasting_engine.predict_next_month with backward-compatible fallbacks."""
    province_df = df[df["Province"] == province].sort_values("Date").copy()

    if pd.isna(next_temp):
        raise ValueError(f"กรุณาระบุ Temperature_C สำหรับ {province}")

    try:
        result = predict_next_month(
            province_df,
            province=None,
            next_temp=float(next_temp),
            alpha=alpha,
            trained_result=trained_result,
        )
    except TypeError:
        try:
            result = predict_next_month(
                province_df,
                next_temp=float(next_temp),
                alpha=alpha,
                trained_result=trained_result,
            )
        except TypeError:
            try:
                result = predict_next_month(
                    province_df,
                    next_temp=float(next_temp),
                    alpha=alpha,
                )
            except TypeError:
                try:
                    result = predict_next_month(province_df, float(next_temp), alpha=alpha)
                except TypeError:
                    result = predict_next_month(province_df)

    if isinstance(result, tuple):
        pred = result[0]
        pred_date = result[1] if len(result) > 1 else None
    elif isinstance(result, dict):
        pred = (
            result.get("prediction")
            or result.get("predicted_load")
            or result.get("Forecast")
        )
        pred_date = result.get("next_date") or result.get("Forecast_Month")
    else:
        pred = result
        pred_date = None

    if pred_date is None:
        _, _, pred_date, _ = get_date_status(province_df)

    return float(pred), pd.to_datetime(pred_date)


def excel_bytes(sheets: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if data is not None and not data.empty:
                data.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()


def build_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty:
        _, latest_month, next_month, _ = get_date_status(df)
    else:
        latest_month, next_month = pd.NaT, pd.NaT

    prediction_history = st.session_state.get("prediction_history", pd.DataFrame())
    if prediction_history is not None and not prediction_history.empty:
        forecast_month = str(prediction_history["Forecast_Month"].iloc[-1])
        provinces_forecasted = int(prediction_history["Province"].nunique())
    else:
        forecast_month = next_month.strftime("%Y-%m") if pd.notna(next_month) else ""
        provinces_forecasted = 0

    rows = [
        ("app_file", APP_FILE),
        ("engine_file", ENGINE_FILE),
        ("model_version", MODEL_VERSION),
        ("target_name", TARGET_DISPLAY_NAME),
        ("target_definition", TARGET_DEFINITION_TH),
        ("primary_data_source", DATA_SOURCE_NAME),
        ("external_validation_sources", "PEA / MEA / EGAT"),
        ("automatic_context_sources", EXTERNAL_CONTEXT_SOURCES),
        ("dataset_latest_month", latest_month.strftime("%Y-%m") if pd.notna(latest_month) else ""),
        ("forecast_month", forecast_month),
        ("run_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("number_of_provinces_forecasted", provinces_forecasted),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def render_section_header(title: str, caption: str = ""):
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div>
                <div class="section-title">{title}</div>
                {caption_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(items):
    cards = "".join(
        f"""
        <div class="status-item">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="status-strip">{cards}</div>', unsafe_allow_html=True)


def render_project_kpis():
    st.markdown(
        """
        <div class="project-meta">
            <div class="project-meta-card">
                <div class="project-meta-icon">🏛️</div>
                <div>
                    <div class="project-meta-label">Institution</div>
                    <div class="project-meta-value">Faculty of Engineering<br>Kasem Bundit University</div>
                </div>
            </div>
            <div class="project-meta-card">
                <div class="project-meta-icon">🎓</div>
                <div>
                    <div class="project-meta-label">Project Type</div>
                    <div class="project-meta-value">Final Year Engineering Project</div>
                </div>
            </div>
            <div class="project-meta-card">
                <div class="project-meta-icon">👤</div>
                <div>
                    <div class="project-meta-label">Developed By</div>
                    <div class="project-meta-value">Tanaphon Bokaew  , Chirachai Phoomkaew, Sutee  Mungkrobklang</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <div class="footer-title">Electricity Demand Forecasting System — E-Report Proxy</div>
            <div class="section-caption">
                Primary target: government electricity consumption reported through Energy e-Report.
                External validation: PEA / MEA / EGAT.
            </div>
            <div class="footer-grid">
                <div>
                    <div class="footer-label">Institution</div>
                    <div class="footer-value">Faculty of Engineering, Kasem Bundit University</div>
                </div>
                <div>
                    <div class="footer-label">Developer</div>
                    <div class="footer-value">Tanaphon Bokaew  , Chirachai  Phoomkaew, Sutee  Mungkrobklang</div>
                </div>
                <div>
                    <div class="footer-label">Workflow</div>
                    <div class="footer-value">Import → Update → Train → Forecast → Validate → Export</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "df": None,
        "uploaded": False,
        "forecasted": False,
        "all_results": None,
        "summary_df": None,
        "prediction_history": pd.DataFrame(),
        "accuracy_history": pd.DataFrame(),
        "external_validation_df": pd.DataFrame(),
        "ai_method_results": {},
        "ai_training_results_df": pd.DataFrame(),
        "ai_training_summary_df": pd.DataFrame(),
        "ai_training_cache_keys": {},
        "ai_accuracy_summary_df": pd.DataFrame(),
        "ai_comparison_df": pd.DataFrame(),
        "uploaded_file_hash": None,
        "uploaded_file_name": None,
        "dataset_source": None,
        "external_validation_editor_version": 0,
        # Start with an empty workspace. Data should only appear after an upload.
        "auto_load_default_dataset": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

default_dataset_path = APP_DIR / "Dataset.xlsx"
if (
    st.session_state.df is None
    and st.session_state.auto_load_default_dataset
    and default_dataset_path.exists()
):
    try:
        default_bytes = default_dataset_path.read_bytes()
        st.session_state.df = clean_input_data(
            pd.read_excel(BytesIO(default_bytes))
        )
        st.session_state.uploaded = True
        st.session_state.uploaded_file_hash = hashlib.md5(default_bytes).hexdigest()
        st.session_state.uploaded_file_name = default_dataset_path.name
        st.session_state.default_dataset_loaded = True
    except Exception as default_load_error:
        st.session_state.default_dataset_loaded = False
        st.session_state.default_dataset_error = str(default_load_error)



# =================================
# TITLE
# =================================



# Banner is retained behind a UI flag so it can be restored without changing assets.
if SHOW_APP_BANNER:
    st.markdown('<div class="hero"><div class="hero-image">', unsafe_allow_html=True)
    st.image(str(APP_DIR / "images" / "S__56598541.jpg"), use_column_width=True)
    st.markdown(
        """
        </div>
        <div class="hero-copy">
            <div class="eyebrow">Electricity Demand Forecasting</div>
            <h1 class="hero-title">ระบบพยากรณ์ความต้องการใช้ไฟฟ้า บริเวณพื้นที่ภาคกลาง</h1>
            <p class="hero-subtitle">
                ใช้ข้อมูล Energy e-Report เป็น Provincial Load Proxy สำหรับการฝึกและพยากรณ์
                พร้อมตรวจสอบทิศทางภายนอกกับข้อมูล PEA, MEA และ EGAT
            </p>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="main-system-title">ระบบพยากรณ์ความต้องการใช้ไฟฟ้า บริเวณพื้นที่ภาคกลาง</div>',
    unsafe_allow_html=True,
)

render_project_kpis()

reset_spacer, reset_col = st.columns([9, 1])
with reset_col:
    if st.button("↻ Reset", help="ล้างข้อมูลและสถานะทั้งหมด", use_container_width=True):
        st.session_state.clear()
        # Without this flag the next rerun immediately reloads Dataset.xlsx,
        # making Reset appear to do nothing.
        st.session_state.auto_load_default_dataset = False
        st.rerun()



# =========================
# SIDEBAR
# =========================
alpha = 1.0
test_ratio = 0.2

# =========================
# TABS
# =========================
main_tab_labels = [
    "◫ Data Import",
    "✦ AI Training",
    "⌁ Forecasting",
    "◎ Accuracy Test",
    "↻ Data Update",
    "▥ Reports",
]
if SHOW_ADVANCED_TOOLS:
    main_tab_labels.extend(["🛡️ Scope & Validation", "🧪 Model Experiment"])

app_tabs = st.tabs(main_tab_labels)
tab_upload, tab_train, tab_predict, tab_accuracy, tab_update, tab_export = app_tabs[:6]
if SHOW_ADVANCED_TOOLS:
    tab_validation, tab_experiment = app_tabs[6:]
else:
    advanced_tab_host = st.empty()
    with advanced_tab_host.container():
        tab_validation, tab_experiment = st.tabs(
            ["🛡️ Scope & Validation", "🧪 Model Experiment"]
        )

# =========================
# TAB 1 UPLOAD
# =========================
with tab_upload:
    if SHOW_TARGET_DEFINITION_NOTICE:
        st.info(
            f"📘 นิยามตัวแปรเป้าหมาย: {TARGET_DEFINITION_TH} "
            "ข้อมูลนี้ไม่ใช่โหลดไฟฟ้ารวมของผู้ใช้ทุกประเภทในจังหวัด"
        )

    uploaded_file = st.file_uploader(
        "เลือกไฟล์ Excel (.xlsx)",
        type=["xlsx"],
        key="external_dataset_upload",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            previous_file_hash = st.session_state.get("uploaded_file_hash")

            raw_df = pd.read_excel(BytesIO(file_bytes))
            df = clean_input_data(raw_df)

            # อ่านไฟล์จริงทุกครั้ง เพื่อกัน session เก่าค้าง
            st.session_state.df = df
            st.session_state.uploaded = True
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.dataset_source = "Dataset ภายนอก"

            if "accuracy_history" not in st.session_state:
                st.session_state.accuracy_history = pd.DataFrame()

            # รีเซ็ตผลเฉพาะเมื่อผู้ใช้เปลี่ยน Dataset จริง ๆ ไม่ใช่ทุก uploader rerun
            if previous_file_hash is not None and previous_file_hash != file_hash:
                st.session_state.forecasted = False
                st.session_state.all_results = None
                st.session_state.summary_df = None
                st.session_state.prediction_history = pd.DataFrame()
                st.session_state.accuracy_history = pd.DataFrame()
                st.session_state.ai_method_results = {}
                st.session_state.ai_training_results_df = pd.DataFrame()
                st.session_state.ai_training_summary_df = pd.DataFrame()
                st.session_state.ai_training_cache_keys = {}
                st.session_state.pop("ai_training_provinces", None)

            current_month, latest_month, next_month, status = get_date_status(df)
            latest_available = latest_month
            default_predict_month = latest_available + pd.DateOffset(months=1)

            if SHOW_ADVANCED_TOOLS:
                target_month = st.date_input(
                    "เลือกเดือนที่ต้องการ Predict",
                    value=default_predict_month,
                    format="YYYY-MM-DD"
                )
            else:
                target_month = default_predict_month

            target_month = pd.Timestamp(target_month).to_period("M").to_timestamp()
            required_month = target_month - pd.DateOffset(months=1)

            if latest_available < required_month:
                st.error(
                    f"ข้อมูลยังไม่ครบ "
                    f"ต้องเติมข้อมูลให้ถึง {required_month.strftime('%Y-%m')} ก่อน "
                    f"จึงจะ Predict เดือน {target_month.strftime('%Y-%m')} ได้"
                )
                can_predict = False
            else:
                st.success(
                    f"✅ ข้อมูลพร้อม Predict เดือน {target_month.strftime('%Y-%m')}"
                )
                can_predict = True
            status_df = latest_status_table(df)
            overall_latest = status_df["Latest_Date"].max()
            complete_until = status_df["Latest_Date"].min()
            need_update = int((status_df["Months_Behind"] > 0).sum())

            st.success("✅ อัปโหลดและเตรียมข้อมูลสำเร็จ")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("ข้อมูลล่าสุดถึง", latest_month.strftime("%Y-%m"))
            c3.metric("เดือนที่จะพยากรณ์", next_month.strftime("%Y-%m"))
            c4.metric("จำนวนจังหวัด", df["Province"].nunique())

            st.info(status)

            st.subheader("📌 สถานะข้อมูลรายจังหวัด")
            k1, k2, k3 = st.columns(3)
            k1.metric("ข้อมูลล่าสุดที่สุดในไฟล์", overall_latest.strftime("%Y-%m"))
            k2.metric("ข้อมูลครบทุกจังหวัดถึง", complete_until.strftime("%Y-%m"))
            k3.metric("จำนวนจังหวัดที่ต้องเติม", need_update)

            st.dataframe(
                status_df[["Province", "Latest_Month", "Months_Behind", "Status"]],
                use_container_width=True,
                height=550,
            )

            

        except Exception as e:
            st.error(f"❌ อัปโหลดไม่สำเร็จ: {e}")

    elif st.session_state.uploaded and st.session_state.df is not None:
        df = st.session_state.df
        current_month, latest_month, next_month, status = get_date_status(df)
        status_df = latest_status_table(df)
        source_name = st.session_state.get("dataset_source") or st.session_state.get("uploaded_file_name") or "Dataset"
        st.success(f"✅ พร้อมใช้งาน: {source_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("ข้อมูลล่าสุดถึง", latest_month.strftime("%Y-%m"))
        c3.metric("เดือนที่จะพยากรณ์", next_month.strftime("%Y-%m"))
        c4.metric("จำนวนจังหวัด", df["Province"].nunique())
        st.info(status)
        st.dataframe(status_df[["Province", "Latest_Month", "Months_Behind", "Status"]], use_container_width=True, height=550)
    else:
        st.warning("กรุณาอัปโหลดไฟล์ Excel ก่อน")

# =========================
# TAB 2 UPDATE DATASET
# =========================
with tab_update:
    render_section_header(
        "Data Update",
        "เพิ่มข้อมูล E-Report Proxy และอุณหภูมิสำหรับเดือนใหม่ก่อนนำไปฝึกโมเดล",
    )



    if not st.session_state.uploaded or st.session_state.df is None:
        st.warning("กรุณาอัปโหลดข้อมูลก่อน")
    else:
        df = st.session_state.df.copy()

        provinces = sorted(df["Province"].dropna().unique().tolist())

        latest_month = df["Date"].max().to_period("M").to_timestamp()
        next_month = latest_month + pd.DateOffset(months=1)

        st.info(
            f"ข้อมูลล่าสุด: {latest_month.strftime('%Y-%m')} | "
            f"เดือนที่จะเพิ่ม: {next_month.strftime('%Y-%m')}"
        )

        target_date = st.date_input(
            "เลือกเดือนที่ต้องการเพิ่มข้อมูล",
            value=next_month,
            format="YYYY-MM-DD"
        )

        target_month = pd.Timestamp(target_date).to_period("M").to_timestamp()
        target_month_text = target_month.strftime("%Y-%m")

        def make_update_template() -> pd.DataFrame:
            return pd.DataFrame({
                "Province": provinces,
                "Date": target_month_text,
                "Load_(kWh)": 0.0,
                "Temperature_C": 0.0,
            })

        # ถ้าเปลี่ยนเดือน หรือ session editor ว่าง/คอลัมน์หาย ให้สร้างตารางใหม่
        existing_update_df = st.session_state.get("update_input_df", pd.DataFrame())
        need_new_update_table = (
            st.session_state.get("update_month") != target_month_text
            or not isinstance(existing_update_df, pd.DataFrame)
            or existing_update_df.empty
            or not {"Province", "Date", "Load_(kWh)", "Temperature_C"}.issubset(existing_update_df.columns)
        )
        if need_new_update_table:
            st.session_state.update_month = target_month_text
            st.session_state.update_input_df = make_update_template()

        if st.button("🌎 ดึง Temperature จาก NASA ทุกจังหวัด", key="nasa_update_all"):
            nasa_rows = []

            for p in provinces:
                lat, lon = PROVINCE_COORDS[p]

                temp = get_nasa_monthly_temp(
                    lat,
                    lon,
                    target_month.year,
                    target_month.month
                )

                nasa_rows.append({
                    "Province": p,
                    "Date": target_month_text,
                    "Load_(kWh)": 0.0,
                    "Temperature_C": round(temp, 2)
                })

            st.session_state.update_input_df = pd.DataFrame(nasa_rows)
            st.success("✅ ดึง Temperature จาก NASA ครบทุกจังหวัดแล้ว")
            st.rerun()

        update_df = st.session_state.update_input_df.copy()
        update_df.columns = [str(c).strip() for c in update_df.columns]

        edited_update_df = st.data_editor(
            update_df,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Province", "Date"],
            key="update_dataset_editor"
        )
        edited_update_df = edited_update_df.copy()
        edited_update_df.columns = [str(c).strip() for c in edited_update_df.columns]

        LOAD_COL = "Load_(kWh)"

        if LOAD_COL not in edited_update_df.columns:
            # กันกรณี editor คืนค่า empty หลัง rerun หรือหลังบันทึก Dataset แล้ว session ค้าง
            st.warning("⚠️ ตารางกรอกข้อมูลถูกรีเซ็ตอัตโนมัติ เพราะไม่พบคอลัมน์ Load_(kWh)")
            st.session_state.update_input_df = make_update_template()
            st.rerun()

        edited_update_df.columns = edited_update_df.columns.str.strip()
        edited_update_df[LOAD_COL] = pd.to_numeric(edited_update_df[LOAD_COL], errors="coerce").fillna(0)
        edited_update_df["Temperature_C"] = pd.to_numeric(edited_update_df["Temperature_C"], errors="coerce").fillna(0)

        filled = int((edited_update_df[LOAD_COL] > 0).sum())
        st.progress(filled / max(len(edited_update_df), 1))
        st.info(f"📊 กรอกแล้ว {filled}/{len(edited_update_df)} จังหวัด")

        if st.button("💾 เพิ่มข้อมูลเข้า Dataset", key="save_update_dataset"):
            try:
                if len(edited_update_df) != EXPECTED_PROVINCE_COUNT:
                    st.error(f"❌ จำนวนจังหวัดไม่ครบ ต้องมี {EXPECTED_PROVINCE_COUNT} จังหวัด แต่พบ {len(edited_update_df)}")
                    st.stop()

                if (edited_update_df["Load_(kWh)"] <= 0).any():
                    st.error("❌ ยังกรอก E-Report Load Proxy ไม่ครบทุกจังหวัด")
                    st.stop()

                if (edited_update_df["Temperature_C"] <= 0).any():
                    st.error("❌ ยังกรอก Temperature_C ไม่ครบทุกจังหวัด")
                    st.stop()

                new_df = edited_update_df.copy()

                new_df["Date"] = pd.to_datetime(new_df["Date"] + "-01")
                new_df["Load_(kWh)"] = pd.to_numeric(new_df["Load_(kWh)"], errors="coerce")
                new_df["Temperature_C"] = pd.to_numeric(new_df["Temperature_C"], errors="coerce")

                # สร้างคอลัมน์ให้เหมือน Dataset เดิม
                new_df["Year"] = new_df["Date"].dt.year
                new_df["Month"] = new_df["Date"].dt.strftime("%b")
                new_df["Month_Num"] = new_df["Date"].dt.month
                

                # เรียงคอลัมน์ให้ตรง Dataset เดิม
                new_df = new_df[
                    [
                        "Province",
                        "Date",
                        "Year",
                        "Month",
                        "Load_(kWh)",
                        "Month_Num",
                        "Temperature_C",
                        
                    ]
                ]

                combined_df = pd.concat([df, new_df], ignore_index=True)

                combined_df = combined_df.drop_duplicates(
                    subset=["Province", "Date"],
                  keep="last"
                )

                # เรียงข้อมูลกลับให้เหมือน Dataset เดิม
                combined_df = combined_df.sort_values(
                    ["Province", "Date"]
                ).reset_index(drop=True)

                dataset_path = APP_DIR / "Dataset.xlsx"
                combined_df.to_excel(dataset_path, index=False)

                st.session_state.df = combined_df
                st.session_state.update_input_df = pd.DataFrame()

                st.success(f"✅ บันทึกข้อมูลเข้า {dataset_path.name} สำเร็จ")
                st.rerun()

            except Exception as e:
                st.error(f"❌ บันทึกไม่สำเร็จ: {e}")

# =========================
# TAB 3 TRAIN
# =========================
with tab_train:
    render_section_header(
        "AI Training",
        "ฝึกโมเดล Provincial Load Proxy ทุกจังหวัดและสรุปตัวชี้วัดประสิทธิภาพ",
    )
    st.caption(
        "Training target: Energy e-Report Provincial Load Proxy | "
        "PEA/MEA/EGAT ใช้เฉพาะ External Validation และไม่ถูกนำมาปนกับข้อมูลฝึก"
    )

    if not st.session_state.uploaded or st.session_state.df is None:
        st.warning("กรุณาอัปโหลดข้อมูลก่อน")
    else:
        df = st.session_state.df
        available_months = sorted(df["Date"].dt.to_period("M").astype(str).unique())

        latest_train_month_text = st.selectbox(
            "เลือกข้อมูลล่าสุดถึง",
            options=available_months[11:],
            index=len(available_months[11:]) - 1
        )

        train_cutoff = pd.Period(latest_train_month_text).to_timestamp()
        forecast_month = train_cutoff + pd.DateOffset(months=1)

        df_train = df[df["Date"] <= train_cutoff].copy()

        st.info(
            f"เลือกข้อมูลล่าสุดถึง {train_cutoff.strftime('%Y-%m')} | "
            f"ระบบจะ Forecast เดือน {forecast_month.strftime('%Y-%m')}"
        )
        _, latest_month, next_month, status = get_date_status(df_train)
        c1, c2, c3 = st.columns(3)
        c1.metric("ข้อมูลล่าสุดถึง", latest_month.strftime("%Y-%m"))
        c2.metric("ระบบจะ Forecast", forecast_month.strftime("%Y-%m"))
        c3.metric("จำนวนจังหวัด", df["Province"].nunique())
        st.info(status)

        st.markdown(
            """
            <div class="ai-console-title">
                <span class="ai-console-icon">🧠</span>
                <strong>AI TRAINING</strong>
            </div>
            <div class="ai-console-subtitle">ฝึกสอนแบบจำลองด้วย AI 3 วิธี โดยจัดเก็บผลแยกตาม Method และ Province</div>
            """,
            unsafe_allow_html=True,
        )
        ai_province_options = sorted(df_train["Province"].dropna().astype(str).unique())
        with st.container(border=True):
            province_col, select_all_col = st.columns([4, 1])
            with select_all_col:
                select_all_ai_training_provinces = st.checkbox(
                    "Select All Provinces",
                    key="ai_training_select_all_provinces",
                )
            if select_all_ai_training_provinces:
                st.session_state["ai_training_provinces"] = ai_province_options
            elif "ai_training_provinces" not in st.session_state:
                st.session_state["ai_training_provinces"] = ai_province_options[:1]
            with province_col:
                ai_selected_provinces = st.multiselect(
                    "เลือกจังหวัดสำหรับ AI Methods",
                    options=ai_province_options,
                    key="ai_training_provinces",
                )

        ai_method_ids = list(AI_METHOD_LABELS)
        selected_training_methods = st.multiselect(
            "เลือกวิธีที่ต้องการ Train",
            options=ai_method_ids,
            default=ai_method_ids,
            format_func=lambda method_id: AI_METHOD_LABELS[method_id],
            key="ai_selected_training_methods",
            help="เลือกได้ 1–3 วิธี แล้วกดปุ่ม Train วิธีที่เลือก",
        )
        requested_method_ids = None

        compare_col, clear_visual_col = st.columns([3, 1])
        if compare_col.button(
            "▶ Train วิธีที่เลือก — Train & Compare Selected",
            key="train_compare_all",
            type="primary",
            use_container_width=True,
            disabled=not selected_training_methods or not ai_selected_provinces,
        ):
            requested_method_ids = selected_training_methods
        clear_visual_col.caption(
            f"เลือก {len(selected_training_methods)}/3 วิธี • "
            f"{len(ai_selected_provinces)}/{len(ai_province_options)} จังหวัด"
        )
        force_retrain_ai = clear_visual_col.checkbox(
            "Train ใหม่แม้มี Cache",
            key="ai_force_retrain",
        )

        method_columns = st.columns(3)
        method_cards = [
            (
                "chatgpt_carr1m", "chatgpt", "◉ ChatGPT (CARR-1M)",
                "Coverage-Aware Residual Reconciliation",
                "ใช้ Seasonal Anchor + Robust Level Adjustment + Ridge/HGB Residual<br>"
                "<b>ตัวแปรนำเข้า</b><br>Load, Temperature, Seasonal/Trend Features",
                "▶ เริ่มฝึกสอน ChatGPT (CARR-1M)", "train_chatgpt_carr1m",
            ),
            (
                "gemini_cg_hae_v1", "gemini", "✦ Gemini (CG-HAE-v1)",
                "Central-GovLoad Hierarchical Adjusted Ensemble",
                "ใช้ STL Seasonal/Trend ร่วมกับ LightGBM และ Reporting Adjustment<br>"
                "<b>ตัวแปรนำเข้า</b><br>Load, Temperature, Calendar และ Reporting (ถ้ามี)",
                "▶ เริ่มฝึกสอน Gemini (CG-HAE-v1)", "train_gemini_cg_hae_v1",
            ),
            (
                "kimi_pgf_tea", "kimi", "K′ Kimi (PGF-TEA)",
                "Provincial Government Forecast Trend-Exponential-Adjustment",
                "ใช้ Seasonal OLS + Temperature CDD + EWMA Recent Adjustment<br>"
                "<b>ตัวแปรนำเข้า</b><br>Load, Temperature และ Reporting Coverage (ถ้ามี)",
                "▶ เริ่มฝึกสอน Kimi (PGF-TEA)", "train_kimi_pgf_tea",
            ),
        ]
        for method_column, card in zip(method_columns, method_cards):
            method_id, css_class, brand, method_name, detail, button_label, button_key = card
            trained_count = len(st.session_state.get("ai_method_results", {}).get(method_id, {}))
            status_text = f"ฝึกแล้ว {trained_count} จังหวัด" if trained_count else "พร้อมฝึกสอน"
            with method_column:
                st.markdown(
                    f"""
                    <div class="method-card {css_class}">
                        <div class="method-brand">{brand}</div>
                        <div class="method-name">{method_name}</div>
                        <span class="method-status">{status_text}</span>
                        <div class="method-detail">{detail}<br><br>
                        <b>การประเมินผล</b><br>MAPE &nbsp; • &nbsp; RMSE &nbsp; • &nbsp; R²</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    button_label,
                    key=button_key,
                    use_container_width=True,
                    disabled=not ai_selected_provinces,
                ):
                    requested_method_ids = [method_id]

        if requested_method_ids is not None:
            if not ai_selected_provinces:
                st.error("กรุณาเลือกอย่างน้อย 1 จังหวัด")
            else:
                labels = ", ".join(AI_METHOD_LABELS[item] for item in requested_method_ids)
                cache_columns = ["Province", "Date", "Load_(kWh)", "Temperature_C"]
                dataset_digest = hashlib.sha256(
                    pd.util.hash_pandas_object(
                        df_train[cache_columns], index=False
                    ).values.tobytes()
                ).hexdigest()
                run_signature = f"{dataset_digest}|{train_cutoff.strftime('%Y-%m-%d')}"
                cache_keys = st.session_state.get("ai_training_cache_keys", {})
                stored_artifacts = st.session_state.get("ai_method_results", {})
                previous_rows = st.session_state.get(
                    "ai_training_results_df", pd.DataFrame()
                )

                pending_by_method = {}
                cached_count = 0
                for method_id in requested_method_ids:
                    pending_provinces = []
                    for province in ai_selected_provinces:
                        cache_id = f"{method_id}|{province}"
                        artifact_exists = province in stored_artifacts.get(method_id, {})
                        result_exists = (
                            not previous_rows.empty
                            and (
                                (previous_rows["Method_ID"] == method_id)
                                & (previous_rows["Province"] == province)
                            ).any()
                        )
                        cache_matches = cache_keys.get(cache_id) == run_signature
                        if not force_retrain_ai and artifact_exists and result_exists and cache_matches:
                            cached_count += 1
                        else:
                            pending_provinces.append(province)
                    pending_by_method[method_id] = pending_provinces

                total_pending = sum(len(items) for items in pending_by_method.values())
                trained_so_far = 0
                progress_bar = st.progress(0.0, text="เตรียม AI Training...")
                new_artifacts = {method_id: {} for method_id in requested_method_ids}
                new_frames = []

                if total_pending:
                    with st.spinner(f"กำลัง Train {labels}..."):
                        for method_id in requested_method_ids:
                            method_provinces = pending_by_method[method_id]
                            if not method_provinces:
                                continue
                            completed_before_method = trained_so_far

                            def update_training_progress(completed, total, active_method, province):
                                overall_completed = completed_before_method + completed
                                progress_bar.progress(
                                    overall_completed / total_pending,
                                    text=(
                                        f"{AI_METHOD_LABELS[active_method]} — {province} "
                                        f"({overall_completed}/{total_pending})"
                                    ),
                                )

                            method_artifacts, method_rows = train_ai_methods(
                                df_train,
                                method_provinces,
                                [method_id],
                                train_cutoff,
                                progress_callback=update_training_progress,
                            )
                            new_artifacts[method_id].update(method_artifacts[method_id])
                            new_frames.append(method_rows)
                            trained_so_far += len(method_provinces)
                    progress_bar.progress(1.0, text="AI Training เสร็จแล้ว")
                    new_rows = pd.concat(new_frames, ignore_index=True)
                else:
                    progress_bar.empty()
                    new_rows = pd.DataFrame()
                    st.info(f"ใช้ผลจาก Cache ครบ {cached_count} รายการ ไม่ต้อง Train ซ้ำ")

                stored_artifacts = st.session_state.get("ai_method_results", {})
                for method_id, province_artifacts in new_artifacts.items():
                    stored_artifacts.setdefault(method_id, {})
                    stored_artifacts[method_id].update(province_artifacts)
                st.session_state.ai_method_results = stored_artifacts

                for method_id, province_artifacts in new_artifacts.items():
                    for province in province_artifacts:
                        cache_keys[f"{method_id}|{province}"] = run_signature
                st.session_state.ai_training_cache_keys = cache_keys

                combined_rows = pd.concat([previous_rows, new_rows], ignore_index=True)
                combined_rows = combined_rows.drop_duplicates(
                    subset=["Method_ID", "Province"], keep="last"
                )
                st.session_state.ai_training_results_df = combined_rows
                st.session_state.ai_training_summary_df = summarize_ai_training(combined_rows)
                if not new_rows.empty:
                    error_count = int((new_rows["Status"] == "ERROR").sum())
                    if error_count:
                        st.error(f"Train เสร็จโดยพบ ERROR {error_count} รายการ")
                    else:
                        st.success(
                            f"Train ใหม่ {len(new_rows)} รายการ | ใช้ Cache {cached_count} รายการ"
                        )

        ai_results_df = st.session_state.get("ai_training_results_df", pd.DataFrame())
        if ai_results_df is not None and not ai_results_df.empty:
            st.markdown("#### Method × Province comparison")
            st.dataframe(
                ai_results_df[
                    ["Method", "Province", "MAPE", "RMSE", "R2", "Status", "Error"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("#### Summary")
            st.dataframe(
                st.session_state.ai_training_summary_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                """
                <div class="training-empty">
                    <div style="font-size:2rem">🗄️</div>
                    <b>ยังไม่มีผลการฝึกสอน</b><br>
                    กรุณาเลือกจังหวัด ช่วงข้อมูล และเริ่มการฝึกสอน
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        if not SHOW_ADVANCED_TOOLS:
            st.markdown(
                '<span class="legacy-ui-hide-from-here"></span>',
                unsafe_allow_html=True,
            )
        st.subheader("⚙️ Legacy Training")

        st.subheader("⚡ One-click Forecast")
        st.caption(
            "ปุ่มเดียวจะ Train ทุกจังหวัด ดึงอุณหภูมิอัตโนมัติ Forecast ครบ 19 จังหวัด "
            "และแนบสถานะ Data Quality โดยไม่ต้องเลือกจังหวัดหรือกรอกทีละช่อง"
        )
        if st.button(
            "⚡ Train + Forecast ทุกจังหวัดอัตโนมัติ",
            type="primary",
            key="one_click_train_forecast_all",
        ):
            with st.spinner(
                "กำลัง Train โมเดล ดึงข้อมูลอุณหภูมิ และพยากรณ์ครบทุกจังหวัด..."
            ):
                try:
                    if df_train["Province"].nunique() != EXPECTED_PROVINCE_COUNT:
                        raise ValueError(
                            f"Dataset ต้องมี {EXPECTED_PROVINCE_COUNT} จังหวัด "
                            f"แต่พบ {df_train['Province'].nunique()} จังหวัด"
                        )

                    all_results = forecast_all_provinces(
                        df_train,
                        test_ratio=test_ratio,
                        alpha=alpha,
                    )
                    summary_df = create_summary_dataframe(all_results)
                    if summary_df is None or summary_df.empty:
                        raise ValueError("ไม่มีผลสรุปจากการ Train")

                    quality_lookup = (
                        build_data_quality_table(df_train)
                        .set_index("Province")["Data_Quality_Status"]
                        .to_dict()
                    )
                    provinces_to_forecast = sorted(df_train["Province"].unique())
                    automatic_temperatures = get_all_automatic_temperatures(
                        df_train,
                        provinces_to_forecast,
                        forecast_month,
                    )
                    automatic_rows = []
                    temperature_fallback_count = 0

                    for province in provinces_to_forecast:
                        trained_result = find_trained_result(all_results, province)
                        if trained_result is None:
                            raise ValueError(f"ไม่พบโมเดลที่ Train แล้วสำหรับ {province}")

                        next_temp, temperature_source = automatic_temperatures[province]
                        if temperature_source != "NASA POWER":
                            temperature_fallback_count += 1

                        pred_kwh, _ = call_predict(
                            df_train,
                            province,
                            next_temp,
                            alpha,
                            trained_result=trained_result,
                        )
                        best_model = (
                            trained_result.get("Best_Model")
                            or trained_result.get("best_model_name")
                            or trained_result.get("model_name", "Unknown")
                        )
                    automatic_rows.append(
                            {
                                "Method_ID": "legacy",
                                "Method_Name": "Legacy Forecasting Engine",
                                "Province": province,
                                "Latest_Month": train_cutoff.strftime("%Y-%m"),
                                "Forecast_Month": forecast_month.strftime("%Y-%m"),
                                "Temperature_C": next_temp,
                                "Temperature_Source": temperature_source,
                                "Best_Model": best_model,
                                "Predicted_Load_kWh": pred_kwh,
                                "Target_Type": TARGET_DISPLAY_NAME,
                                "Primary_Source": DATA_SOURCE_NAME,
                                "Data_Quality_Flag": quality_lookup.get(
                                    province, "⚠️ ไม่พบสถานะ"
                                ),
                                "External_Context": "PEA / MEA / EGAT direction check",
                                "Status": "One-click Forecast สำเร็จ",
                                "Run_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )

                    automatic_result_df = pd.DataFrame(automatic_rows)
                    previous_history = st.session_state.get(
                        "prediction_history", pd.DataFrame()
                    )
                    combined_history = merge_prediction_history(
                        previous_history, automatic_result_df
                    )

                    st.session_state.train_df = df_train
                    st.session_state.target_forecast_month = forecast_month.strftime("%Y-%m")
                    st.session_state.all_results = all_results
                    st.session_state.summary_df = summary_df
                    st.session_state.prediction_history = combined_history
                    st.session_state.forecasted = True
                    st.session_state.last_model_file_hash = st.session_state.get(
                        "uploaded_file_hash"
                    )

                    st.success(
                        f"✅ พยากรณ์สำเร็จครบ {len(automatic_result_df)} จังหวัด "
                        f"สำหรับเดือน {forecast_month.strftime('%Y-%m')}"
                    )
                    if temperature_fallback_count:
                        st.info(
                            f"NASA ยังไม่มีข้อมูลบางพื้นที่/ช่วงเวลา "
                            f"ระบบใช้ค่ากลางอุณหภูมิเดือนเดียวกันย้อนหลังแทน "
                            f"{temperature_fallback_count} จังหวัด"
                        )
                    st.dataframe(
                        automatic_result_df,
                        use_container_width=True,
                        height=560,
                        hide_index=True,
                    )
                except Exception as e:
                    st.error(f"❌ One-click Forecast ไม่สำเร็จ: {e}")

        st.divider()
        st.caption(
            "โหมดทดลองแบบทีละขั้น: Train โมเดลก่อน "
            "จากนั้นจึงเลือกจังหวัด ปรับข้อมูลนำเข้า และ Forecast ด้านล่าง"
        )
        if st.button("🧪 Train โมเดลทุกจังหวัด (โหมดทดลอง)", type="primary"):
            with st.spinner("กำลัง Train โมเดล..."):
                try:
                    all_results = forecast_all_provinces(df_train, test_ratio=test_ratio, alpha=alpha)
                    st.session_state.train_df = df_train
                    st.session_state.target_forecast_month = forecast_month.strftime("%Y-%m")
                    summary_df = create_summary_dataframe(all_results)
                    if summary_df is None or summary_df.empty:
                        st.error("❌ Train ไม่สำเร็จ: ไม่มีผลลัพธ์จากโมเดล")
                        st.write("all_results:", all_results)
                        st.stop()
                    st.session_state.all_results = all_results
                    st.session_state.summary_df = summary_df
                    st.session_state.forecasted = True
                    st.session_state.last_model_file_hash = st.session_state.get("uploaded_file_hash")
                    st.success("✅ Train โมเดลครบทุกจังหวัดสำเร็จ")
                except Exception as e:
                    st.error(f"❌ Train โมเดลไม่สำเร็จ: {e}")

        if st.session_state.forecasted and st.session_state.summary_df is not None:
            summary_df = st.session_state.summary_df
            st.subheader("📋 สรุปผลการ Train โมเดล")
            st.dataframe(summary_df, use_container_width=True, height=550)
            mape_col = find_metric_col(summary_df, "MAPE")
            rmse_col = find_metric_col(summary_df, "RMSE")
            r2_col = find_metric_col(summary_df, "R2")
            if mape_col and rmse_col and r2_col:
                k1, k2, k3 = st.columns(3)
                k1.metric("MAPE เฉลี่ย", f"{summary_df[mape_col].mean():.2f}%")
                k2.metric("RMSE เฉลี่ย", f"{summary_df[rmse_col].mean():.2f}")
                k3.metric("R² เฉลี่ย", f"{summary_df[r2_col].mean():.4f}")

# =========================
# TAB 4 PREDICT
# =========================
with tab_predict:
    render_section_header(
        "Forecasting",
        "เลือกจังหวัด ระบุอุณหภูมิ และพยากรณ์ E-Report Provincial Load Proxy เดือนถัดไป",
    )

    ai_store = st.session_state.get("ai_method_results", {})
    trained_ai_methods = [
        method_id for method_id in AI_METHOD_LABELS if ai_store.get(method_id)
    ]
    if trained_ai_methods and st.session_state.df is not None:
        st.subheader("🤖 AI Method Forecasting")
        selected_ai_methods = st.multiselect(
            "Forecasting Method",
            options=trained_ai_methods,
            default=trained_ai_methods,
            format_func=lambda method_id: AI_METHOD_LABELS[method_id],
            key="ai_forecast_methods",
        )
        trained_provinces = sorted({
            province
            for method_id in selected_ai_methods
            for province in ai_store.get(method_id, {})
        })
        select_all_ai_provinces = st.checkbox(
            "Select All Provinces",
            key="ai_forecast_select_all_provinces",
        )
        if select_all_ai_provinces:
            st.session_state["ai_forecast_provinces"] = trained_provinces
        elif "ai_forecast_provinces" not in st.session_state:
            st.session_state["ai_forecast_provinces"] = trained_provinces[:1]
        selected_ai_provinces = st.multiselect(
            "จังหวัดสำหรับ AI Forecast",
            options=trained_provinces,
            key="ai_forecast_provinces",
        )
        selection_signature = tuple(selected_ai_provinces)
        if st.session_state.get("ai_forecast_selection_signature") != selection_signature:
            st.session_state.ai_forecast_selection_signature = selection_signature
            st.session_state.ai_temperature_fetch_message = None
        temperature_rows = []
        for province in selected_ai_provinces:
            available_artifacts = [
                ai_store[method_id].get(province) for method_id in selected_ai_methods
                if ai_store.get(method_id, {}).get(province) is not None
            ]
            forecast_text = ""
            if available_artifacts:
                cutoff = pd.Timestamp(available_artifacts[0]["train_cutoff"])
                forecast_text = (cutoff + pd.DateOffset(months=1)).strftime("%Y-%m")
            temperature_rows.append({
                "Province": province,
                "Forecast_Month": forecast_text,
                "Temperature_C": None,
                "Temperature_Source": "รอดึงข้อมูล",
                "Status": "รอดำเนินการ",
            })

        temperature_template = pd.DataFrame(temperature_rows)
        stored_temperature_df = st.session_state.get(
            "ai_forecast_temperature_df", pd.DataFrame()
        )
        if not stored_temperature_df.empty and not temperature_template.empty:
            preserved_columns = [
                "Province", "Forecast_Month", "Temperature_C",
                "Temperature_Source", "Status",
            ]
            stored_temperature_df = stored_temperature_df[
                [column for column in preserved_columns if column in stored_temperature_df.columns]
            ]
            temperature_template = temperature_template.drop(
                columns=["Temperature_C", "Temperature_Source", "Status"]
            ).merge(
                stored_temperature_df,
                on=["Province", "Forecast_Month"],
                how="left",
            )
            temperature_template["Temperature_Source"] = (
                temperature_template["Temperature_Source"].fillna("รอดึงข้อมูล")
            )
            temperature_template["Status"] = (
                temperature_template["Status"].fillna("รอดำเนินการ")
            )
        st.session_state.ai_forecast_temperature_df = temperature_template

        if st.button(
            "🌡️ Get NASA Temperature for Selected Provinces",
            key="get_ai_nasa_temperatures",
        ):
            try:
                if not selected_ai_provinces:
                    raise ValueError("กรุณาเลือก Province อย่างน้อย 1 จังหวัด")
                forecast_months = temperature_template["Forecast_Month"].dropna().unique()
                if len(forecast_months) != 1 or not str(forecast_months[0]):
                    raise ValueError("Trained models ต้องมี Forecast_Month เดียวกัน")
                forecast_month = pd.Timestamp(f"{forecast_months[0]}-01")
                automatic_temperatures = get_all_automatic_temperatures(
                    st.session_state.df,
                    selected_ai_provinces,
                    forecast_month,
                )
                populated_rows = []
                for row in temperature_rows:
                    temperature_value, temperature_source = automatic_temperatures[
                        row["Province"]
                    ]
                    row["Temperature_C"] = temperature_value
                    row["Temperature_Source"] = temperature_source
                    row["Status"] = (
                        "NASA SUCCESS"
                        if temperature_source == "NASA POWER"
                        else "FALLBACK USED"
                    )
                    populated_rows.append(row)
                st.session_state.ai_forecast_temperature_df = pd.DataFrame(populated_rows)
                st.session_state.ai_forecast_temperature_editor_version = (
                    st.session_state.get("ai_forecast_temperature_editor_version", 0) + 1
                )
                fallback_count = sum(
                    source != "NASA POWER"
                    for _, source in automatic_temperatures.values()
                )
                st.session_state.ai_temperature_fetch_message = (
                    f"Temperature พร้อม {len(populated_rows)}/{len(selected_ai_provinces)} จังหวัด | "
                    f"NASA {len(populated_rows) - fallback_count} | Fallback {fallback_count}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"ดึง Temperature ไม่สำเร็จ: {exc}")

        if st.session_state.get("ai_temperature_fetch_message"):
            st.success(st.session_state.ai_temperature_fetch_message)
        edited_ai_temperatures = st.data_editor(
            st.session_state.ai_forecast_temperature_df,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Province", "Forecast_Month", "Temperature_Source", "Status"],
            key=(
                "ai_forecast_temperature_editor_"
                f"{st.session_state.get('ai_forecast_temperature_editor_version', 0)}"
            ),
        )
        st.session_state.ai_forecast_temperature_df = edited_ai_temperatures.copy()
        if st.button("🔮 Forecast Selected AI Methods", key="predict_ai_methods"):
            try:
                if not selected_ai_methods or not selected_ai_provinces:
                    raise ValueError("กรุณาเลือก Method และ Province")
                if edited_ai_temperatures["Temperature_C"].isna().any():
                    raise ValueError("กรุณาระบุ Temperature_C ให้ครบ")
                temperature_map = {
                    row["Province"]: float(row["Temperature_C"])
                    for _, row in edited_ai_temperatures.iterrows()
                }
                ai_predictions = predict_ai_methods(
                    st.session_state.df,
                    ai_store,
                    selected_ai_methods,
                    selected_ai_provinces,
                    temperature_map,
                )
                st.session_state.prediction_history = merge_prediction_history(
                    st.session_state.prediction_history,
                    ai_predictions,
                )
                st.success(f"AI Forecast สำเร็จ {len(ai_predictions)} รายการ")
                st.dataframe(ai_predictions, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"AI Forecast ไม่สำเร็จ: {exc}")
        st.divider()
        if not SHOW_ADVANCED_TOOLS:
            st.markdown(
                '<span class="legacy-ui-hide-from-here"></span>',
                unsafe_allow_html=True,
            )
        st.subheader("⚙️ Legacy Forecasting")

    if not st.session_state.uploaded or st.session_state.df is None:
        st.warning("กรุณาอัปโหลดข้อมูลก่อน")

    elif not st.session_state.forecasted:
        st.info("Legacy Forecasting: กรุณากด Legacy Training ก่อน")

    else:
        df = st.session_state.get("train_df", st.session_state.df)
        target_forecast_month = st.session_state.get("target_forecast_month", None)
        if target_forecast_month:
            forced_forecast_month = pd.to_datetime(target_forecast_month + "-01")
            forced_latest_month = forced_forecast_month - pd.DateOffset(months=1)
        else:
            forced_latest_month = df["Date"].max().to_period("M").to_timestamp()
            forced_forecast_month = forced_latest_month + pd.DateOffset(months=1)

        provinces = sorted(df["Province"].dropna().unique().tolist())

        selected_provinces = st.multiselect(
            "เลือกจังหวัดที่ต้องการ Predict",
            options=provinces,
            default=[provinces[0]] if provinces else []
        )

        select_all_provinces = st.checkbox("เลือกทุกจังหวัด")
        if select_all_provinces:
            selected_provinces = provinces
            if len(selected_provinces) != EXPECTED_PROVINCE_COUNT:
                st.warning(
                    f"ต้องมีจังหวัดครบ {EXPECTED_PROVINCE_COUNT} จังหวัดก่อน Predict ทุกจังหวัด "
                    f"แต่พบ {len(selected_provinces)} จังหวัด กรุณาตรวจสอบ Dataset"
                )
                st.stop()

        if not selected_provinces:
            st.warning("กรุณาเลือกอย่างน้อย 1 จังหวัด")

        else:
            input_rows = []

            for province in selected_provinces:
                province_df = df[df["Province"] == province].sort_values("Date")

                latest_month = forced_latest_month
                next_month = forced_forecast_month

                default_temp = float(province_df["Temperature_C"].iloc[-1])

                input_rows.append({
                    "Province": province,
                    "Latest_Month": forced_latest_month.strftime("%Y-%m"),
                    "Forecast_Month": forced_forecast_month.strftime("%Y-%m"),
                    "Temperature_C": None,
                    "Status": "รออุณหภูมิ"
                })

            input_df = pd.DataFrame(input_rows)

            st.subheader("🌡️ ตารางอุณหภูมิรายจังหวัด")
            st.caption("แก้ไขอุณหภูมิได้ก่อนกด Predict")

            input_df = pd.DataFrame(input_rows)

            if "predict_input_df" not in st.session_state:
                st.session_state.predict_input_df = pd.DataFrame(input_rows)
            else:
                old_provinces = st.session_state.predict_input_df["Province"].tolist()
                new_provinces = [r["Province"] for r in input_rows]

                old_months = st.session_state.predict_input_df["Forecast_Month"].astype(str).tolist()
                new_months = [r["Forecast_Month"] for r in input_rows]

                if old_provinces != new_provinces or old_months != new_months:
                    st.session_state.predict_input_df = pd.DataFrame(input_rows)

            edited_df = st.data_editor(
                st.session_state.predict_input_df,
                use_container_width=True,
                num_rows="fixed",
                disabled=["Province", "Latest_Month", "Forecast_Month", "Status"]
            )
             
            if st.button("🌍 ดึงอุณหภูมิจาก NASA"):
                try:
                    temp_df = st.session_state.predict_input_df.copy()

                    for idx, row in temp_df.iterrows():
                        province = row["Province"]
                        forecast_month = pd.to_datetime(row["Forecast_Month"])

                        lat, lon = PROVINCE_COORDS[province]

                        nasa_temp = get_nasa_monthly_temp(
                            lat,
                            lon,
                            forecast_month.year,
                            forecast_month.month
                        )

                        temp_df.at[idx, "Temperature_C"] = round(nasa_temp, 2)
                        temp_df.at[idx, "Status"] = "ดึงจาก NASA แล้ว"

                    st.session_state.predict_input_df = temp_df
                    st.success("✅ ดึงอุณหภูมิจาก NASA สำเร็จ")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ ดึง NASA ไม่สำเร็จ: {e}")

                    if edited_df["Temperature_C"].isna().any():
                        st.warning("กรุณากรอกอุณหภูมิให้ครบ หรือกดดึงจาก NASA")
                        st.stop()

            if st.button("🔮 Predict จังหวัดที่เลือก", type="primary"):
                try:
                    result_rows = []

                    for _, row in edited_df.iterrows():
                        province = row["Province"]
                        if pd.isna(row["Temperature_C"]):
                            raise ValueError(f"กรุณาระบุ Temperature_C สำหรับ {province}")
                        next_temp = float(row["Temperature_C"])

                        province_df = df[df["Province"] == province].sort_values("Date")

                        latest_month = forced_latest_month
                        next_month = forced_forecast_month

                        # ค้นหา trained_result จากรายการผลลัพธ์
                        trained_result = None
                        if st.session_state.all_results is not None:
                            # all_results เป็น list ของ dict
                            if isinstance(st.session_state.all_results, list):
                                for result in st.session_state.all_results:
                                    if result.get("Province") == province:
                                        trained_result = result
                                        break
                            # หรือเป็น dict ที่ keyed by province
                            elif isinstance(st.session_state.all_results, dict):
                                trained_result = st.session_state.all_results.get(province)
                        
                        if trained_result is None:
                            raise ValueError(f"ไม่พบผลโมเดลที่ train แล้วสำหรับ {province} กรุณากด Train / Forecast ใหม่")

                        pred_kwh, pred_date = call_predict(
                            df,
                            province,
                            next_temp,
                            alpha,
                            trained_result=trained_result,
                        )
                        province_df = df[df["Province"] == province].copy()

                        best_model = trained_result.get("Best_Model") or trained_result.get("best_model_name") or trained_result.get("model_name", "Unknown")

                        result_rows.append({
                            "Method_ID": "legacy",
                            "Method_Name": "Legacy Forecasting Engine",
                            "Province": province,
                            "Latest_Month": latest_month.strftime("%Y-%m"),
                            "Forecast_Month": next_month.strftime("%Y-%m"),
                            "Temperature_C": next_temp,
                            "Best_Model": best_model,
                            "Predicted_Load_kWh": pred_kwh,
                            "Status": "Predict สำเร็จ",
                            "Run_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    result_df = pd.DataFrame(result_rows)

                    st.success("✅ Predict สำเร็จ")
                    st.dataframe(result_df, use_container_width=True, height=550)

                    st.subheader("📊 ตรวจสอบความแม่นยำ Actual Load (E-Report Proxy)")

                    accuracy_df = result_df.copy()

                    accuracy_df["Actual_Load_kWh"] = None
                    accuracy_df["Error_%"] = None
                    accuracy_df["Accuracy_Status"] = "รอกรอก Actual"

                    edited_accuracy_df = st.data_editor(
                        accuracy_df,
                        use_container_width=True,
                        num_rows="fixed",
                        disabled=[
                            "Province",
                            "Latest_Month",
                            "Forecast_Month",
                            "Temperature_C",
                           
                            "Predicted_Load_kWh",
                            "Status",
                            "Run_At",
                            "Error_%",
                            "Accuracy_Status",
                        ]
                    )

                    if st.button("📌 คำนวณความแม่นยำ"):
                        try:
                            check_df = edited_accuracy_df.copy()

                            check_df["Actual_Load_kWh"] = pd.to_numeric(
                                check_df["Actual_Load_kWh"],
                                errors="coerce"
                            )

                            check_df["Error_%"] = (
                                abs(check_df["Actual_Load_kWh"] - check_df["Predicted_Load_kWh"])
                                / check_df["Actual_Load_kWh"]
                                * 100
                            )

                            check_df["Accuracy_Status"] = check_df["Error_%"].apply(
                                lambda x: "✅ PASS ≤ 10%" if pd.notna(x) and x <= 10 else "❌ FAIL > 10%"
                            )

                            st.success("✅ คำนวณความแม่นยำสำเร็จ")
                            st.dataframe(check_df, use_container_width=True, height=550)

                            checked_inline_df = check_df.dropna(subset=["Actual_Load_kWh", "Predicted_Load_kWh", "Error_%"])
                            if not checked_inline_df.empty:
                                rmse_inline = np.sqrt(
                                    np.mean(
                                        (checked_inline_df["Actual_Load_kWh"] - checked_inline_df["Predicted_Load_kWh"]) ** 2
                                    )
                                )
                                m1, m2, m3 = st.columns(3)
                                m1.metric("MAPE", f"{checked_inline_df['Error_%'].mean():.2f}%")
                                m2.metric("RMSE", f"{rmse_inline:,.2f} kWh")
                                m3.metric("ผ่าน ≤10%", int((checked_inline_df["Error_%"] <= 10).sum()))

                            st.session_state.accuracy_history = pd.concat(
                                [st.session_state.get("accuracy_history", pd.DataFrame()), check_df],
                                ignore_index=True
                            )

                        except Exception as e:
                            st.error(f"❌ คำนวณไม่สำเร็จ: {e}")

                    # เก็บผลล่าสุดเพียง 1 แถวต่อจังหวัดและเดือนพยากรณ์
                    # ป้องกันกรณีผู้ใช้กด One-click Forecast แล้วกด Predict ซ้ำ
                    st.session_state.prediction_history = merge_prediction_history(
                        st.session_state.prediction_history, result_df
                    )

                except Exception as e:
                    st.error(f"❌ Predict ไม่สำเร็จ: {e}")

        if not st.session_state.prediction_history.empty:
            st.subheader("📌 ประวัติการ Predict")
            st.dataframe(st.session_state.prediction_history, use_container_width=True, height=550)
# ==========================
# TAB 5 ACCURACY CHECK
# ==========================
with tab_accuracy:
    # ทำความสะอาดประวัติเดิมที่อาจถูกบันทึกซ้ำก่อนเพิ่มการป้องกันนี้
    if not st.session_state.prediction_history.empty:
        history_df = st.session_state.prediction_history.copy()
        if {"Province", "Forecast_Month"}.issubset(history_df.columns):
            st.session_state.prediction_history = merge_prediction_history(
                pd.DataFrame(), history_df
            )

    render_section_header(
        "Accuracy Check",
        "อัปโหลด Actual Load (E-Report Proxy) เพื่อเทียบกับผลพยากรณ์และบันทึกประวัติ",
    )

    st.info(
        "อัปโหลด Actual จาก Energy e-Report เพื่อเทียบกับผล Predict "
        "คะแนนนี้วัดความแม่นยำภายในขอบเขต Proxy ไม่ใช่โหลดไฟฟ้ารวมของจังหวัด"
    )

    if st.session_state.prediction_history.empty:
        st.warning("ยังไม่มี Prediction History กรุณา Predict ก่อน")
    else:
        st.subheader("📌 Prediction History")
        st.dataframe(st.session_state.prediction_history, use_container_width=True, height=550)

        # =========================
        # ตัวอย่างไฟล์ Actual
        # =========================

        sample_actual = (
            st.session_state.prediction_history[["Province", "Forecast_Month"]]
            .drop_duplicates()
            .rename(columns={"Forecast_Month": "Date"})
            .sort_values(["Province", "Date"])
            .reset_index(drop=True)
        )

        sample_actual["Actual_Load_kWh"] = ""
        sample_actual = sample_actual[["Province", "Date", "Actual_Load_kWh"]]

        # Keep the generated sample data for the downloadable template,
        # but do not render the large preview table in the normal workflow.

        # =========================
        # Download Template
        # =========================
        from io import BytesIO

        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            sample_actual.to_excel(writer, index=False)

        # =========================
        # Upload Actual
        # =========================
        actual_file = st.file_uploader(
            "อัปโหลดไฟล์ Actual Excel (.xlsx)",
            type=["xlsx"],
            help="ต้องมี Province, Date, Actual_Load_kWh",
            key="actual_file_upload"
        )

        st.download_button(
            label="⬇️ Download Template Actual Excel",
            data=buffer.getvalue(),
            file_name="Actual_Load_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if actual_file is not None:
            try:
                actual_df = pd.read_excel(actual_file)
                merged_df, accuracy_summary, comparison_df = calculate_method_accuracy(
                    st.session_state.prediction_history,
                    actual_df,
                )
                st.session_state.ai_accuracy_summary_df = accuracy_summary
                st.session_state.ai_comparison_df = comparison_df
                st.success("✅ คำนวณ Method-aware Accuracy สำเร็จ")
                st.dataframe(merged_df, use_container_width=True, height=650)
                if not accuracy_summary.empty:
                    st.subheader("AI Comparison Summary")
                    st.dataframe(accuracy_summary, use_container_width=True, hide_index=True)
                if not comparison_df.empty:
                    st.subheader("AI Comparison by Province")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                if st.button("💾 Save Accuracy History"):
                    combined_accuracy = pd.concat(
                        [st.session_state.accuracy_history, merged_df],
                        ignore_index=True,
                    )
                    st.session_state.accuracy_history = combined_accuracy.drop_duplicates(
                        ["Method_ID", "Province", "Forecast_Month"], keep="last"
                    ).reset_index(drop=True)
                    st.success("✅ บันทึก Accuracy History สำเร็จ")

            except Exception as e:
                st.error(f"❌ ตรวจ Accuracy ไม่สำเร็จ: {e}")

    if not st.session_state.accuracy_history.empty:
        st.subheader("📚 Accuracy History")
        st.dataframe(st.session_state.accuracy_history, use_container_width=True, height=650)


# =========================================
# TAB 6 DATA SCOPE & EXTERNAL VALIDATION
# =========================================
with tab_validation:
    render_section_header(
        "Data Scope & External Validation",
        "กำหนดขอบเขต E-Report Proxy ตรวจ Data Quality และยืนยันแนวโน้มกับ PEA / MEA / EGAT",
    )

    st.subheader("1) นิยามเชิงปฏิบัติการ")
    st.success(
        f"**{TARGET_DISPLAY_NAME}:** {TARGET_DEFINITION_TH}"
    )
    st.warning(
        "ข้อมูลนี้ไม่เท่ากับโหลดไฟฟ้ารวมของบ้านอยู่อาศัย ธุรกิจ และอุตสาหกรรมทั้งหมด "
        "ผลพยากรณ์จึงตีความภายในขอบเขตหน่วยงานภาครัฐที่รายงานเข้าสู่ e-Report"
    )
    st.dataframe(build_data_definition_table(), use_container_width=True, hide_index=True)

    st.subheader("2) Data Quality Gate ก่อน Train / Forecast")
    if not st.session_state.uploaded or st.session_state.df is None:
        st.info("อัปโหลด Dataset ก่อนเพื่อคำนวณความเปลี่ยนแปลงเดือนต่อเดือนรายจังหวัด")
        quality_df = pd.DataFrame()
    else:
        quality_df = build_data_quality_table(st.session_state.df)
        flagged = quality_df[
            quality_df["MoM_Change_%"].abs() > OUTLIER_THRESHOLD_PCT
        ]
        q1, q2, q3 = st.columns(3)
        q1.metric("จังหวัดทั้งหมด", quality_df["Province"].nunique())
        q2.metric(
            f"ต้องตรวจ Coverage (> ±{OUTLIER_THRESHOLD_PCT:.0f}%)",
            len(flagged),
        )
        q3.metric("ผ่านเกณฑ์เบื้องต้น", len(quality_df) - len(flagged))
        if not flagged.empty:
            st.warning(
                "พบจังหวัดที่เปลี่ยนแปลงรุนแรง: ต้องตรวจจำนวนหน่วยงาน รอบบิล "
                "และสถานะข้อมูลก่อนนำ Actual เดือนล่าสุดไปอัปเดตโมเดล"
            )
        st.dataframe(
            quality_df,
            use_container_width=True,
            height=520,
            column_config={
                "Date": st.column_config.DateColumn("Latest Month", format="YYYY-MM"),
                "Previous_Load_kWh": st.column_config.NumberColumn(
                    "Previous Proxy (kWh)", format="%,.0f"
                ),
                "Load_(kWh)": st.column_config.NumberColumn(
                    "Latest Proxy (kWh)", format="%,.0f"
                ),
                "MoM_Change_%": st.column_config.NumberColumn(
                    "MoM Change", format="%.2f%%"
                ),
            },
            hide_index=True,
        )

    st.subheader("3) External Validation — ตรวจทิศทาง ไม่ใช้ฝึกโมเดล")
    st.caption(
        "กรอกอัตราเปลี่ยนแปลงจากรายงานทางการในช่วงเวลาเดียวกับ Proxy "
        "ระบบจะตรวจว่าเพิ่ม/ลดไปในทิศทางเดียวกันหรือไม่ โดยไม่รวมข้อมูลนี้ใน Training Dataset"
    )
    st.markdown(
        """
        - **PEA:** [รายงานสถานการณ์การจำหน่ายไฟฟ้า](https://www.pea.co.th/about-pea/sales-report)
        - **MEA:** [MEA Open Data](https://opendata.mea.or.th/)
        - **EGAT:** [สถิติความต้องการไฟฟ้า](https://www.egat.co.th/home/statistics-demand-latest/)
        """
    )

    if st.session_state.external_validation_df.empty:
        st.session_state.external_validation_df = pd.DataFrame(
            [
                {
                    "Source": "PEA",
                    "Scope": "16 จังหวัดในพื้นที่ PEA / หรือเขตที่เกี่ยวข้อง",
                    "Reference_Period": "",
                    "Official_Change_%": None,
                    "Proxy_Change_%": None,
                    "Direction_Check": "รอกรอกข้อมูล",
                    "Evidence_URL": "https://www.pea.co.th/about-pea/sales-report",
                    "Note": "ใช้ตรวจแนวโน้มหน่วยจำหน่ายไฟฟ้า",
                },
                {
                    "Source": "MEA",
                    "Scope": "Bangkok, Nonthaburi, Samut Prakan",
                    "Reference_Period": "",
                    "Official_Change_%": None,
                    "Proxy_Change_%": None,
                    "Direction_Check": "รอกรอกข้อมูล",
                    "Evidence_URL": "https://opendata.mea.or.th/",
                    "Note": "ใช้หน่วยขายจำแนกตามจังหวัด",
                },
                {
                    "Source": "EGAT",
                    "Scope": "ระบบไฟฟ้าระดับประเทศ",
                    "Reference_Period": "",
                    "Official_Change_%": None,
                    "Proxy_Change_%": None,
                    "Direction_Check": "รอกรอกข้อมูล",
                    "Evidence_URL": "https://www.egat.co.th/home/statistics-demand-latest/",
                    "Note": "ใช้เป็นภาพรวม ไม่ใช่ Actual รายจังหวัด",
                },
            ]
        )

    if st.button(
        "🧮 คำนวณ E-Report Proxy จาก Dataset อัตโนมัติ",
        key="calculate_external_proxy",
        disabled=not st.session_state.uploaded or st.session_state.df is None,
    ):
        source_df = st.session_state.df.copy()
        source_df["Date"] = pd.to_datetime(source_df["Date"], errors="coerce")
        source_df = source_df.dropna(subset=["Date", "Province", "Load_(kWh)"])
        available_months = sorted(source_df["Date"].dt.to_period("M").unique())

        if len(available_months) < 2:
            st.error("คำนวณไม่ได้: Dataset ต้องมีข้อมูลอย่างน้อย 2 เดือน")
        else:
            previous_period, latest_period = available_months[-2:]
            source_df["Period"] = source_df["Date"].dt.to_period("M")
            mea_provinces = {"Bangkok", "Nonthaburi", "Samut Prakan"}

            def scoped_proxy_change(source):
                if source == "MEA":
                    scoped = source_df[source_df["Province"].isin(mea_provinces)]
                elif source == "PEA":
                    scoped = source_df[~source_df["Province"].isin(mea_provinces)]
                else:  # EGAT is a national context; use the full 19-province proxy.
                    scoped = source_df

                totals = scoped.groupby("Period")["Load_(kWh)"].sum()
                previous_value = totals.get(previous_period, np.nan)
                latest_value = totals.get(latest_period, np.nan)
                if pd.isna(previous_value) or pd.isna(latest_value) or previous_value == 0:
                    return np.nan
                return ((latest_value / previous_value) - 1) * 100

            calculated = st.session_state.external_validation_df.copy()
            calculated["Reference_Period"] = (
                f"{previous_period.strftime('%Y-%m')} → {latest_period.strftime('%Y-%m')}"
            )
            calculated["Proxy_Change_%"] = calculated["Source"].map(scoped_proxy_change)
            calculated["Direction_Check"] = "รอกรอกค่าทางการช่วงเวลาเดียวกัน"
            calculated["Note"] = calculated.apply(
                lambda row: (
                    str(row["Note"]).split(" | Proxy คำนวณอัตโนมัติ")[0]
                    + " | Proxy คำนวณอัตโนมัติจาก Dataset"
                ),
                axis=1,
            )
            st.session_state.external_validation_df = calculated
            st.session_state.external_validation_editor_version += 1
            st.success(
                "คำนวณ E-Report Proxy สำเร็จ: MEA ใช้ Bangkok/Nonthaburi/"
                "Samut Prakan, PEA ใช้จังหวัดที่เหลือ และ EGAT ใช้ภาพรวม 19 จังหวัด"
            )

    validation_input = st.data_editor(
        st.session_state.external_validation_df,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Source", "Scope", "Direction_Check", "Evidence_URL"],
        column_config={
            "Official_Change_%": st.column_config.NumberColumn(
                "Official Change (%)", format="%.2f"
            ),
            "Proxy_Change_%": st.column_config.NumberColumn(
                "E-Report Proxy Change (%)", format="%.2f"
            ),
        },
        key=(
            "external_validation_editor_"
            f"{st.session_state.external_validation_editor_version}"
        ),
    )

    if st.button("🔎 ตรวจ External Validation", key="check_external_validation"):
        checked_validation = validation_input.copy()

        def direction_check(row):
            official = pd.to_numeric(row["Official_Change_%"], errors="coerce")
            proxy = pd.to_numeric(row["Proxy_Change_%"], errors="coerce")
            if pd.isna(official) or pd.isna(proxy):
                return "⚠️ ข้อมูลไม่ครบ"
            if official == 0 or proxy == 0:
                return "🟡 ใกล้ศูนย์ / ต้องอธิบายเพิ่ม"
            return "✅ ทิศทางสอดคล้อง" if np.sign(official) == np.sign(proxy) else "❌ ทิศทางไม่สอดคล้อง"

        checked_validation["Direction_Check"] = checked_validation.apply(
            direction_check, axis=1
        )
        st.session_state.external_validation_df = checked_validation
        st.dataframe(checked_validation, use_container_width=True, hide_index=True)
        st.success(
            "บันทึกผล External Validation แล้ว "
            "ผลนี้ใช้สนับสนุนความสมเหตุสมผลและไม่เปลี่ยนค่าพยากรณ์ของโมเดล"
        )


# =========================
# TAB 7 MODEL EXPERIMENT
# =========================
with tab_experiment:
    render_section_header(
        "Model Experiment — Feature Ablation",
        "เปรียบเทียบตัวแปร 5 รูปแบบด้วย Ridge Regression และ Walk-forward Validation เงื่อนไขเดียวกัน",
    )

    st.info(
        "ข้อมูลต้นทางและช่วงทดสอบเหมือนกันทุกชุด เปลี่ยนเฉพาะกลุ่มตัวแปร "
        "เพื่อให้ผลต่างสะท้อนผลของตัวแปร ไม่ใช่ผลจากการเปลี่ยนอัลกอริทึม"
    )

    experiment_descriptions = {
        "Baseline": "ข้อมูลพื้นฐาน + ค่าโหลดย้อนหลัง + ฤดูกาล + อุณหภูมิ",
        "Calendar": "Baseline + วันทำงาน วันหยุด และเทศกาล",
        "EUI_SEU": "Baseline + EUI/SEU จากเดือนก่อน (lag 1)",
        "Grid_PEA_MEA_EGAT": "Baseline + PEA/MEA/EGAT จากเดือนก่อน (lag 1)",
        "All_Factors": "รวมตัวแปรทุกกลุ่ม",
    }

    selected_variant = st.selectbox(
        "เลือกชุดตัวแปรที่ต้องการตรวจรายละเอียด",
        options=list(experiment_descriptions),
        format_func=lambda value: f"{value} — {experiment_descriptions[value]}",
        key="experiment_variant_selector",
    )

    if st.button("▶️ โหลดและตรวจสอบผลทดลองทั้ง 5 รูปแบบ", key="load_model_experiment"):
        st.session_state.show_model_experiment = True

    if not st.session_state.get("show_model_experiment", False):
        st.caption("กดปุ่มเพื่ออ่านผล Controlled Experiment ที่บันทึกไว้ในไฟล์ Excel")
    elif not EXPERIMENT_FILE.exists():
        st.error(f"ไม่พบไฟล์ผลทดลอง: {EXPERIMENT_FILE}")
    else:
        try:
            overall = pd.read_excel(EXPERIMENT_FILE, sheet_name="Result_Overall")
            monthly = pd.read_excel(EXPERIMENT_FILE, sheet_name="Result_Monthly")
            province = pd.read_excel(EXPERIMENT_FILE, sheet_name="Result_Province")
            detail = pd.read_excel(EXPERIMENT_FILE, sheet_name="Test_Detail")

            overall = overall.loc[:, ~overall.columns.astype(str).str.startswith("Unnamed")]
            overall = overall.loc[:, ~overall.columns.astype(str).str.endswith(".1")]
            monthly["Date"] = pd.to_datetime(monthly["Date"], errors="coerce")
            detail["Date"] = pd.to_datetime(detail["Date"], errors="coerce")

            expected_variants = set(experiment_descriptions)
            found_variants = set(overall["Variant"].dropna().astype(str))
            valid_count = len(detail) == 570
            valid_provinces = detail["Province"].nunique() == 19
            valid_months = detail["Date"].dt.to_period("M").nunique() == 6
            valid_variants = expected_variants == found_variants

            gate_cols = st.columns(4)
            gate_cols[0].metric("ชุดตัวแปร", f"{len(found_variants)}/5")
            gate_cols[1].metric("จังหวัด", f"{detail['Province'].nunique()}/19")
            gate_cols[2].metric("เดือนทดสอบ", f"{detail['Date'].dt.to_period('M').nunique()}/6")
            gate_cols[3].metric("ค่าพยากรณ์", f"{len(detail)}/570")

            if all([valid_count, valid_provinces, valid_months, valid_variants]):
                st.success(
                    "✅ ตรวจสอบหลักฐานครบ: 5 รูปแบบ × 19 จังหวัด × 6 เดือน = 570 ค่าพยากรณ์"
                )
            else:
                st.warning("⚠️ จำนวนข้อมูลไม่ตรงกับแบบแผนการทดลอง กรุณาตรวจไฟล์ผลทดลอง")

            baseline_row = overall.loc[overall["Variant"] == "Baseline"].iloc[0]
            selected_row = overall.loc[overall["Variant"] == selected_variant].iloc[0]
            best_row = overall.loc[overall["MAPE_pct"].idxmin()]

            metric_cols = st.columns(4)
            metric_cols[0].metric("ชุดที่เลือก", selected_variant)
            metric_cols[1].metric(
                "MAPE",
                f"{selected_row['MAPE_pct']:.2f}%",
                f"{selected_row['MAPE_Improvement_vs_Baseline_pct']:+.2f}% vs Baseline",
            )
            metric_cols[2].metric("WAPE", f"{selected_row['WAPE_pct']:.2f}%")
            metric_cols[3].metric("ชุดที่ MAPE ดีที่สุด", str(best_row["Variant"]))

            st.subheader("1) เปรียบเทียบผลรวมทั้ง 5 รูปแบบ")
            overall_display = overall[
                [
                    "Variant", "Forecast_Count", "MAPE_pct", "WAPE_pct",
                    "RMSE_kWh", "Bias_pct", "MAPE_Improvement_vs_Baseline_pct",
                ]
            ].sort_values("MAPE_pct")
            st.dataframe(
                overall_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Forecast_Count": st.column_config.NumberColumn("จำนวน", format="%.0f"),
                    "MAPE_pct": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                    "WAPE_pct": st.column_config.NumberColumn("WAPE", format="%.2f%%"),
                    "RMSE_kWh": st.column_config.NumberColumn("RMSE (kWh)", format="%,.0f"),
                    "Bias_pct": st.column_config.NumberColumn("Bias", format="%.2f%%"),
                    "MAPE_Improvement_vs_Baseline_pct": st.column_config.NumberColumn(
                        "ดีขึ้นเทียบ Baseline", format="%+.2f%%"
                    ),
                },
            )
            st.bar_chart(overall_display.set_index("Variant")[["MAPE_pct", "WAPE_pct"]])

            st.subheader(f"2) ผลรายเดือน — {selected_variant}")
            selected_monthly = monthly.loc[monthly["Variant"] == selected_variant].copy()
            selected_monthly["Month"] = selected_monthly["Date"].dt.strftime("%Y-%m")
            st.dataframe(
                selected_monthly[["Month", "Forecast_Count", "MAPE_pct", "WAPE_pct", "RMSE_kWh", "Bias_pct"]],
                use_container_width=True,
                hide_index=True,
            )
            st.line_chart(selected_monthly.set_index("Month")[["MAPE_pct", "WAPE_pct"]])

            st.subheader(f"3) ผลแยก 19 จังหวัด — {selected_variant}")
            selected_province = province.loc[province["Variant"] == selected_variant].copy()
            selected_province = selected_province.sort_values("MAPE_pct")
            st.dataframe(
                selected_province,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAPE_pct": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                    "WAPE_pct": st.column_config.NumberColumn("WAPE", format="%.2f%%"),
                    "RMSE_kWh": st.column_config.NumberColumn("RMSE (kWh)", format="%,.0f"),
                    "Bias_pct": st.column_config.NumberColumn("Bias", format="%.2f%%"),
                },
            )

            with st.expander("🔍 เปิดหลักฐานค่าจริงและค่าพยากรณ์ทีละรายการ"):
                selected_detail = detail.loc[detail["Variant"] == selected_variant].copy()
                selected_detail["Date"] = selected_detail["Date"].dt.strftime("%Y-%m")
                st.dataframe(selected_detail, use_container_width=True, hide_index=True, height=500)

            st.caption(
                f"โมเดลควบคุม: Ridge Regression + RobustScaler | "
                f"วิธีประเมิน: one-step expanding walk-forward | "
                f"Baseline MAPE {baseline_row['MAPE_pct']:.2f}%"
            )
        except Exception as experiment_error:
            st.error(f"อ่านผลการทดลองไม่สำเร็จ: {experiment_error}")


# =========================
# TAB 8 EXPORT
# =========================
if not SHOW_ADVANCED_TOOLS:
    advanced_tab_host.empty()

with tab_export:
    render_section_header(
        "Export Report",
        "ดาวน์โหลดชุดข้อมูล ผลสรุป และประวัติการพยากรณ์เป็นไฟล์ Excel",
    )

    if not st.session_state.uploaded:
        st.warning("กรุณาอัปโหลดข้อมูลก่อน")
    else:
        ai_model_summary = st.session_state.get(
            "ai_training_results_df", pd.DataFrame()
        ).copy()
        if not ai_model_summary.empty:
            ai_model_summary["Method_Name"] = ai_model_summary["Method"]
        legacy_model_summary = (
            st.session_state.summary_df.copy()
            if st.session_state.summary_df is not None
            else pd.DataFrame()
        )
        if not legacy_model_summary.empty:
            legacy_model_summary["Method_ID"] = "legacy"
            legacy_model_summary["Method_Name"] = "Legacy Forecasting Engine"
        model_summary_export = pd.concat(
            [legacy_model_summary, ai_model_summary], ignore_index=True, sort=False
        )
        sheets = {
            "Input_Data": st.session_state.df,
            "Prediction_History": st.session_state.prediction_history,
            "Accuracy_History": st.session_state.accuracy_history,
            "AI_Comparison": st.session_state.get("ai_comparison_df", pd.DataFrame()),
            "AI_Accuracy_Summary": st.session_state.get(
                "ai_accuracy_summary_df", pd.DataFrame()
            ),
            "Model_Summary": model_summary_export,
            "Data_Definition": build_data_definition_table(),
            "Data_Quality": (
                build_data_quality_table(st.session_state.df)
                if st.session_state.df is not None
                else pd.DataFrame()
            ),
            "External_Validation": st.session_state.external_validation_df,
            "Run_Metadata": build_run_metadata(st.session_state.df),
        }

        st.markdown("### Report Preview")
        preview_sheets = {
            sheet_name: sheet_df
            for sheet_name, sheet_df in sheets.items()
            if sheet_df is not None and not sheet_df.empty
        }
        report_metric_cols = st.columns(4)
        report_metric_cols[0].metric("รายงานที่แสดงได้", len(preview_sheets))
        report_metric_cols[1].metric(
            "Prediction Records",
            len(st.session_state.prediction_history),
        )
        report_metric_cols[2].metric(
            "Accuracy Records",
            len(st.session_state.accuracy_history),
        )
        report_metric_cols[3].metric(
            "Provinces",
            st.session_state.df["Province"].nunique()
            if "Province" in st.session_state.df.columns
            else 0,
        )

        selected_report_name = st.selectbox(
            "เลือก Report ที่ต้องการแสดง",
            options=list(preview_sheets),
            key="report_preview_sheet",
        )
        selected_report_df = preview_sheets[selected_report_name]
        st.caption(
            f"{selected_report_name} — {len(selected_report_df):,} records × "
            f"{len(selected_report_df.columns):,} columns"
        )
        st.dataframe(
            selected_report_df,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        file_bytes = excel_bytes(sheets)
        st.download_button(
            label="⇩ Download Excel Report",
            data=file_bytes,
            file_name=f"electricity_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

render_footer()
