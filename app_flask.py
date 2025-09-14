# -*- coding: utf-8 -*-
"""
CSV Analyzer — Apple-like Flask Web App (Single File, Plotly Interactive)
=========================================================================
- Upload CSV (auto detect encoding & separator, optional manual override)
- Clean Apple-like UI with light/dark mode, cards, plenty of whitespace
- Dataset summary (shape, dtypes, memory, missing, duplicates)
- Preview with pagination + pandas.query() row filter
- Advanced controls (dynamic charts):
    * Histograms (multi)
    * Bar charts (categorical)
    * Box & Violin (numeric distributions)
    * Scatter (x vs y, trendline)
    * Correlation heatmap
    * Time series (choose date column + numeric y + resampling D/W/M/Q/Y)
- Interactive charts powered by Plotly.js
- Export reports (CSV / JSON / Excel) + download all as ZIP
- Session-based (per-user) without DB
- Defensive reading & validation with clear errors
"""

# -----------------------------
# Imports
# -----------------------------
import os
import io
import re
import json
import zipfile
import base64
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from flask import (
    Flask, render_template_string, request, redirect, url_for, flash,
    send_from_directory, session, jsonify, make_response
)

# Optional robust encoding detection
try:
    import chardet
except Exception:
    chardet = None

# -----------------------------
# Configuration
# -----------------------------
APP_NAME = "CSV Analyzer — Clean & Dynamic (Plotly)"
STATIC_DIR = Path("static")
REPORT_DIR = STATIC_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("CSV_ANALYZER_SECRET", "apple-like-secret-key")

# -----------------------------
# Apple-like HTML + CSS (Bootstrap-only layout, custom look & feel)
# -----------------------------
HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ app_name }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap (grid + helpers) -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

  <!-- Plotly -->
  <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>

  <style>
    :root {
      --bg: #f5f5f7;
      --card: #ffffff;
      --text: #1d1d1f;
      --muted: #6e6e73;
      --accent: #0071e3;
      --border: #e5e5ea;
      --shadow: 0 12px 30px rgba(0,0,0,0.09);
    }
    [data-theme="dark"] {
      --bg: #000000;
      --card: #0c0c0c;
      --text: #f5f5f7;
      --muted: #a1a1a6;
      --accent: #0a84ff;
      --border: #2c2c2e;
      --shadow: 0 18px 40px rgba(0,0,0,0.6);
    }
    html, body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      scroll-behavior: smooth;
    }
    .hero {
      text-align: center;
      padding: 48px 16px 12px;
    }
    .hero h1 {
      font-weight: 700;
      letter-spacing: -0.02em;
      font-size: clamp(26px, 3.0vw, 44px);
      margin-bottom: 6px;
    }
    .hero p.lead {
      color: var(--muted);
      font-size: clamp(14px, 1.6vw, 18px);
      margin: 0 auto;
      max-width: 820px;
    }
    .badges .badge {
      background: var(--border);
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 10px;
      font-weight: 500;
    }
    .container-narrow {
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 16px 56px;
    }
    .cardx {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 22px;
      margin-bottom: 24px;
    }
    .btn-primary {
      background: var(--accent);
      border: none;
      border-radius: 28px;
      padding: 10px 20px;
      font-weight: 600;
    }
    .btn-outline {
      border: 1px solid var(--border);
      color: var(--text);
      background: transparent;
      border-radius: 28px;
      padding: 8px 18px;
    }
    .btn-primary:hover { filter: brightness(0.96); }
    .btn-outline:hover { background: var(--border); }
    pre {
      background: rgba(0,0,0,0.04);
      border-radius: 16px;
      padding: 16px;
      color: var(--text);
      border: 1px solid var(--border);
      white-space: pre-wrap;
    }
    .plot-card {
      border-radius: 16px;
      padding: 8px;
      background: var(--card);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .theme-toggle {
      cursor: pointer;
      user-select: none;
      font-weight: 600;
      color: var(--muted);
    }
    .table thead { background: rgba(0,0,0,0.03); }
    .small-muted { color: var(--muted); font-size: 12px; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:var(--border); color:var(--muted); }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 992px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body data-theme="{{ 'dark' if theme=='dark' else 'light' }}">

  <div class="hero">
    <h1>{{ app_name }}</h1>
    <p class="lead">Clean, simple and elegant — Upload a CSV and analyze it interactively with Plotly, smart defaults and downloadable reports.</p>
    <div class="badges mt-2">
      <span class="badge">Pandas</span>
      <span class="badge">Plotly</span>
      <span class="badge">Flask</span>
      <span class="badge">Apple-like UI</span>
    </div>
  </div>

  <div class="container-narrow">
    <!-- Upload & top controls -->
    <div class="cardx">
      <div class="topbar">
        <form method="post" enctype="multipart/form-data" class="w-100">
          <div class="row g-2 align-items-center">
            <div class="col-12 col-lg-5">
              <input class="form-control" type="file" name="file" accept=".csv" required>
            </div>
            <div class="col-6 col-lg-2">
              <input class="form-control" type="text" name="sep" placeholder="Separator (auto)">
            </div>
            <div class="col-6 col-lg-2">
              <input class="form-control" type="text" name="encoding" placeholder="Encoding (auto)">
            </div>
            <div class="col-6 col-lg-2">
              <input class="form-control" type="number" name="head" placeholder="Head" value="8" min="1">
            </div>
            <div class="col-6 col-lg-1 d-grid">
              <button class="btn btn-primary">Analyze</button>
            </div>
          </div>
        </form>
        <div class="theme-toggle" id="themeToggle">🌓 Theme</div>
      </div>
      {% with messages = get_flashed_messages() %}
        {% if messages %}{% for msg in messages %}
          <div class="alert alert-info mt-3 mb-0">{{ msg }}</div>
        {% endfor %}{% endif %}
      {% endwith %}
    </div>

    {% if info %}
    <!-- Dataset info & download center -->
    <div class="cardx">
      <div class="d-flex justify-content-between align-items-center">
        <h3 class="mb-0">Dataset Info</h3>
        <div class="small-muted">{{ filename }}</div>
      </div>
      <pre class="mt-3">{{ info }}</pre>

      <div class="grid-2">
        <div>
          <h5>Quick Filters</h5>
          <form method="get" class="row g-2">
            <input type="hidden" name="sid" value="{{ sid }}">
            <div class="col-12 col-md-8">
              <input class="form-control" name="q" placeholder="pandas.query example: amount < 0 and category == 'Rent'" value="{{ q or '' }}">
              <div class="small-muted mt-1">Pro tip: strings must be in quotes, use and/or, ==, !=, &gt;, &lt;, isin([...])</div>
            </div>
            <div class="col-6 col-md-2">
              <input class="form-control" type="number" name="page" min="1" value="{{ page }}">
            </div>
            <div class="col-6 col-md-2 d-grid">
              <button class="btn btn-outline">Apply</button>
            </div>
          </form>
        </div>
        <div>
          <h5>Downloads</h5>
          {% if files %}
            <ul class="mb-2">
              {% for f in files %}
                <li><a href="{{ url_for('download_file', filename=f) }}" target="_blank">{{ f }}</a></li>
              {% endfor %}
            </ul>
          {% else %}
            <div class="small-muted">No reports yet.</div>
          {% endif %}
          {% if zip_name %}
            <a class="btn btn-outline" href="{{ url_for('download_file', filename=zip_name) }}" target="_blank">⬇️ Download All (ZIP)</a>
          {% endif %}
        </div>
      </div>
    </div>
    {% endif %}

    {% if preview %}
    <!-- Preview table with pagination -->
    <div class="cardx">
      <h3>Preview</h3>
      {{ preview|safe }}
      <div class="d-flex justify-content-between align-items-center mt-2">
        <div class="small-muted">Page {{ page }}</div>
        <div class="d-flex gap-1">
          <a class="btn btn-outline" href="{{ url_for('index', sid=sid, q=q, page=1) }}">⏮️ First</a>
          <a class="btn btn-outline" href="{{ url_for('index', sid=sid, q=q, page=prev_page) }}">◀️ Prev</a>
          <a class="btn btn-outline" href="{{ url_for('index', sid=sid, q=q, page=next_page) }}">Next ▶️</a>
        </div>
      </div>
    </div>
    {% endif %}

    {% if stats_num or stats_cat or missing %}
    <!-- Statistics -->
    <div class="cardx">
      <h3>Statistics</h3>
      <div class="grid-2">
        <div>
          {% if stats_num %}
            <h5>Numeric</h5>
            {{ stats_num|safe }}
          {% endif %}
          {% if missing %}
            <h5 class="mt-3">Missing</h5>
            {{ missing|safe }}
          {% endif %}
        </div>
        <div>
          {% if stats_cat %}
            <h5>Categorical</h5>
            {{ stats_cat|safe }}
          {% endif %}
        </div>
      </div>
    </div>
    {% endif %}

    {% if controls %}
    <!-- Visualization Controls -->
    <div class="cardx">
      <h3>Visualization Controls</h3>
      {{ controls|safe }}
    </div>
    {% endif %}

    {% if plots %}
    <!-- Interactive plots -->
    <div class="cardx">
      <h3>Visualizations</h3>
      <div class="row g-3">
        {% for div in plots %}
          <div class="col-12">
            <div class="plot-card">
              {{ div|safe }}
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if errors %}
    <div class="cardx">
      <h3>Errors</h3>
      <pre class="mb-0">{{ errors }}</pre>
    </div>
    {% endif %}

  </div>

  <script>
    // Theme toggler persisted in localStorage
    const root = document.body;
    const btn = document.getElementById('themeToggle');
    const saved = localStorage.getItem('theme');
    if(saved){ root.setAttribute('data-theme', saved); }
    btn?.addEventListener('click', () => {
      const cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', cur);
      localStorage.setItem('theme', cur);
    });
  </script>
</body>
</html>
"""

# -----------------------------
# Session State
# -----------------------------
@dataclass
class SessionState:
    filename: str = "uploaded.csv"
    df: Optional[pd.DataFrame] = None
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    date_cols: List[str] = field(default_factory=list)
    last_reports: List[str] = field(default_factory=list)
    last_zip: Optional[str] = None
    head_rows: int = 8

SESSIONS: Dict[str, SessionState] = {}

# -----------------------------
# Helpers — I/O detection & parsing
# -----------------------------
COMMON_SEPS = [",", ";", "\t", "|"]

def detect_encoding_sample(flike) -> Optional[str]:
    """Best-effort encoding detection using chardet on a chunk."""
    if chardet is None:
        return None
    pos = flike.tell()
    chunk = flike.read(100_000)
    flike.seek(pos)
    if isinstance(chunk, bytes):
        raw = chunk
    else:
        raw = chunk.encode("utf-8", errors="ignore")
    res = chardet.detect(raw)
    return res.get("encoding")

def detect_sep_sample(flike, encoding="utf-8") -> str:
    """Guess separator by counting occurrences on the first line."""
    pos = flike.tell()
    head = flike.read(4096)
    flike.seek(pos)
    if isinstance(head, bytes):
        try:
            head = head.decode(encoding, errors="ignore")
        except Exception:
            head = head.decode("utf-8", errors="ignore")
    best_sep, best_cols = ",", -1
    first = head.splitlines()[0] if head else ""
    for s in COMMON_SEPS:
        c = first.count(s)
        if c > best_cols:
            best_cols = c; best_sep = s
    return best_sep

def read_csv_defensively(file_storage, sep: Optional[str], encoding: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    """
    Read CSV robustly. If sep/encoding not provided, try to detect.
    Returns (df, used_sep, used_encoding).
    """
    stream = file_storage.stream
    enc = encoding or detect_encoding_sample(stream) or "utf-8"
    sep_used = sep or detect_sep_sample(stream, enc)
    df = pd.read_csv(stream, sep=sep_used, encoding=enc)
    return df, sep_used, enc

# -----------------------------
# Helpers — Data transformation & summaries
# -----------------------------
def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(index=True).sum()) / (1024 ** 2)

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to infer dates and numeric types safely.
    (coerce) ensures invalid values become NaT/NaN instead of raising.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna().astype(str).head(20)
            if any(re.search(r"\d{4}-\d{1,2}-\d{1,2}", s) for s in sample):
                try:
                    out[col] = pd.to_datetime(out[col], errors="coerce")
                except Exception:
                    pass
    for col in out.columns:
        if out[col].dtype == object:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            except Exception:
                pass
    return out

def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dates = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dates]
    return num, cat, dates

def df_summary_text(df: pd.DataFrame, filename: str, used_sep: str, used_encoding: str) -> str:
    parts = [
        f"File: {filename}",
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]}",
        f"Separator: {repr(used_sep)} | Encoding: {used_encoding}",
        f"Memory: {memory_usage_mb(df):.2f} MB",
        "\nDtypes:\n" + str(df.dtypes)
    ]
    na_counts = df.isna().sum()
    if na_counts.any():
        parts.append("\nMissing values (top):\n" + str(na_counts.sort_values(ascending=False).head(20)))
    dup = df.duplicated().sum()
    parts.append(f"\nDuplicate rows: {dup}")
    return "\n".join(parts)

def render_table_html(df: pd.DataFrame, page: int = 1, per_page: int = 20) -> str:
    start = max((page - 1) * per_page, 0)
    end = start + per_page
    chunk = df.iloc[start:end]
    if chunk.empty and page > 1:
        chunk = df.iloc[0:per_page]
    return chunk.to_html(classes="table table-hover table-sm", index=False)

def df_numeric_stats(df: pd.DataFrame) -> Optional[str]:
    num = df.select_dtypes(include=[np.number])
    if num.empty: return None
    return num.describe().T.to_html(classes="table table-bordered table-sm")

def df_categorical_stats(df: pd.DataFrame) -> Optional[str]:
    cat = df.select_dtypes(exclude=[np.number])
    if cat.empty: return None
    return cat.describe().T.to_html(classes="table table-bordered table-sm")

def df_missing_table(df: pd.DataFrame) -> Optional[str]:
    mis = df.isna().sum()
    if not mis.any(): return None
    return mis.to_frame("missing_count").to_html(classes="table table-bordered table-sm")

# -----------------------------
# Helpers — Exports
# -----------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def export_reports(df: pd.DataFrame, sid: str, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Generate CSV/JSON/Excel reports and return the list of file names.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = []

    # Head CSV
    head_name = f"{sid}_head_{ts}.csv"
    df.head(100).to_csv(REPORT_DIR / head_name, index=False)
    names.append(head_name)

    # Full JSON (sampled if huge)
    js_name = f"{sid}_data_{ts}.json"
    if len(df) > 20000:
        df.sample(20000, random_state=42).to_json(REPORT_DIR / js_name, orient="records", force_ascii=False)
    else:
        df.to_json(REPORT_DIR / js_name, orient="records", force_ascii=False)
    names.append(js_name)

    # Excel workbook with multiple sheets
    xl_name = f"{sid}_report_{ts}.xlsx"
    with pd.ExcelWriter(REPORT_DIR / xl_name, engine="openpyxl") as xw:
        df.head(1000).to_excel(xw, sheet_name="data_head", index=False)
        if numeric_cols:
            df[numeric_cols].describe().T.to_excel(xw, sheet_name="numeric_stats")
        if categorical_cols:
            df[categorical_cols].describe().T.to_excel(xw, sheet_name="categorical_stats")
        mis = df.isna().sum()
        if mis.any():
            mis.to_frame("missing_count").to_excel(xw, sheet_name="missing")
        if len(numeric_cols) >= 2:
            df[numeric_cols].corr().to_excel(xw, sheet_name="correlation")
    names.append(xl_name)

    return names

def make_zip(files: List[str], sid: str) -> Optional[str]:
    if not files: return None
    zip_name = f"{sid}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = REPORT_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            fpath = REPORT_DIR / f
            if fpath.exists():
                z.write(fpath, arcname=f)
    return zip_name

# -----------------------------
# Plotly chart builders (safe with Series.squeeze())
# -----------------------------
def p_histogram(df: pd.DataFrame, col: str, nbins: int = 30, title: Optional[str] = None) -> str:
    import plotly.express as px
    if col not in df.columns:
        return f"<div class='small-muted'>Column {col} not found.</div>"

    sub = df[col].dropna().squeeze()
    if sub.empty:
        return f"<div class='small-muted'>No data for histogram: {col}</div>"

    fig = px.histogram(sub, x=col, nbins=nbins, title=title or f"Histogram — {col}",
                       color_discrete_sequence=["#0071e3"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_bar_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> str:
    import plotly.express as px
    if col not in df.columns:
        return f"<div class='small-muted'>Column {col} not found.</div>"

    vc = df[col].astype(str).value_counts().head(top_n)
    if vc.empty:
        return f"<div class='small-muted'>No data for bar chart: {col}</div>"

    fig = px.bar(vc.reset_index(), x="index", y=col, title=f"Bar — {col}",
                 color_discrete_sequence=["#0071e3"])
    fig.update_layout(xaxis_title=col, yaxis_title="Count",
                      margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_box(df: pd.DataFrame, cols: List[str]) -> str:
    import plotly.express as px
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna()
    if sub.empty:
        return "<div class='small-muted'>No data for boxplot</div>"

    m = sub.melt(var_name="variable", value_name="value")
    fig = px.box(m, x="variable", y="value", title="Boxplot — Outliers",
                 color_discrete_sequence=["#0071e3"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_violin(df: pd.DataFrame, cols: List[str]) -> str:
    import plotly.express as px
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna()
    if sub.empty:
        return "<div class='small-muted'>No data for violin plot</div>"

    m = sub.melt(var_name="variable", value_name="value")
    fig = px.violin(m, x="variable", y="value", box=True, points="suspectedoutliers",
                    title="Violin — Distribution",
                    color_discrete_sequence=["#0071e3"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_scatter(df: pd.DataFrame, x: str, y: str, trendline: bool = True) -> str:
    import plotly.express as px
    if x not in df.columns or y not in df.columns:
        return "<div class='small-muted'>Scatter needs valid columns.</div>"

    sub = df[[x, y]].dropna()
    if sub.empty:
        return "<div class='small-muted'>Scatter needs numeric X and Y with data.</div>"

    x_series = sub[x].squeeze()
    y_series = sub[y].squeeze()

    if not np.issubdtype(x_series.dtype, np.number) or not np.issubdtype(y_series.dtype, np.number):
        return "<div class='small-muted'>Scatter needs numeric X and Y with data.</div>"

    fig = px.scatter(sub, x=x, y=y, opacity=0.8,
                     trendline="ols" if trendline else None,
                     color_discrete_sequence=["#0071e3"],
                     title=f"Scatter — {x} vs {y}")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_heatmap(df: pd.DataFrame, cols: List[str]) -> str:
    import plotly.express as px
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return "<div class='small-muted'>Heatmap needs at least 2 numeric columns.</div>"

    sub = df[cols].dropna()
    if sub.empty:
        return "<div class='small-muted'>No data for heatmap</div>"

    corr = sub.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def p_timeseries(df: pd.DataFrame, date_col: str, y_col: str, freq: str = "M") -> str:
    import plotly.express as px
    if date_col not in df.columns or y_col not in df.columns:
        return "<div class='small-muted'>Time series needs a valid date column and numeric Y.</div>"

    ser = df[[date_col, y_col]].dropna()
    if ser.empty:
        return "<div class='small-muted'>No data for time series</div>"

    date_series = ser[date_col].squeeze()
    y_series = ser[y_col].squeeze()

    # Ensure datetime
    if not np.issubdtype(date_series.dtype, np.datetime64):
        try:
            date_series = pd.to_datetime(date_series, errors="coerce")
        except Exception:
            return "<div class='small-muted'>Failed to parse datetime.</div>"

    ser = pd.DataFrame({date_col: date_series, y_col: y_series}).dropna()
    if ser.empty:
        return "<div class='small-muted'>No parsable dates.</div>"

    ser = ser.set_index(date_col).resample(freq).sum(numeric_only=True).reset_index()
    if ser.empty:
        return "<div class='small-muted'>No data after resampling.</div>"

    fig = px.line(ser, x=date_col, y=y_col, markers=True,
                  title=f"Time Series — {y_col} ({freq})",
                  color_discrete_sequence=["#0071e3"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="simple_white")
    return fig.to_html(include_plotlyjs=False, full_html=False)

# -----------------------------
# Forms — Controls (HTML)
# -----------------------------
def render_controls_form(state: "SessionState") -> str:
    num_opts = "".join(f'<option value="{c}">{c}</option>' for c in state.numeric_cols)
    cat_opts = "".join(f'<option value="{c}">{c}</option>' for c in state.categorical_cols)
    date_opts = "".join(f'<option value="{c}">{c}</option>' for c in state.date_cols)

    return f"""
    <form method="post" action="{url_for('make_plots')}" class="row g-3">
      <input type="hidden" name="sid" value="{get_sid()}">

      <div class="col-12 col-xl-6">
        <div class="cardx">
          <h5>Distributions</h5>
          <div class="row g-2">
            <div class="col-12">
              <label class="form-label">Histograms (numeric, multiple)</label>
              <select multiple class="form-select" name="hist_cols">{num_opts}</select>
              <div class="small-muted">Hold CTRL/Cmd to select multiple</div>
            </div>
            <div class="col-6">
              <label class="form-label">Boxplot (numeric, multiple)</label>
              <select multiple class="form-select" name="box_cols">{num_opts}</select>
            </div>
            <div class="col-6">
              <label class="form-label">Violin (numeric, multiple)</label>
              <select multiple class="form-select" name="violin_cols">{num_opts}</select>
            </div>
            <div class="col-12">
              <label class="form-label">Bar (categorical)</label>
              <select class="form-select" name="bar_col">{cat_opts}</select>
            </div>
          </div>
        </div>
      </div>

      <div class="col-12 col-xl-6">
        <div class="cardx">
          <h5>Relationships</h5>
          <div class="row g-2">
            <div class="col-6">
              <label class="form-label">Scatter X (numeric)</label>
              <select class="form-select" name="scatter_x">{num_opts}</select>
            </div>
            <div class="col-6">
              <label class="form-label">Scatter Y (numeric)</label>
              <select class="form-select" name="scatter_y">{num_opts}</select>
            </div>
            <div class="col-12">
              <div class="form-check mt-2">
                <input class="form-check-input" type="checkbox" name="trendline" id="trendline" checked>
                <label class="form-check-label" for="trendline">Trendline (OLS)</label>
              </div>
            </div>
          </div>

          <h5 class="mt-3">Time Series</h5>
          <div class="row g-2">
            <div class="col-5">
              <label class="form-label">Date column</label>
              <select class="form-select" name="date_col">{date_opts}</select>
            </div>
            <div class="col-5">
              <label class="form-label">Y (numeric)</label>
              <select class="form-select" name="y_col">{num_opts}</select>
            </div>
            <div class="col-2">
              <label class="form-label">Freq</label>
              <select class="form-select" name="freq">
                <option value="D">D</option><option value="W">W</option>
                <option value="M" selected>M</option><option value="Q">Q</option><option value="Y">Y</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div class="col-12 d-flex gap-2">
        <button class="btn btn-primary" type="submit">Generate Plots</button>
        <a class="btn btn-outline" href="{url_for('reset_session')}">Reset Session</a>
      </div>
    </form>
    """

# -----------------------------
# Flask utils
# -----------------------------
def get_sid() -> str:
    if "sid" not in session:
        session["sid"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return session["sid"]

def get_state() -> SessionState:
    sid = get_sid()
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return SESSIONS[sid]

def clear_state():
    sid = get_sid()
    if sid in SESSIONS:
        del SESSIONS[sid]

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET: show controls, current previews, stats, plots if session has df (with pagination/filter)
    POST: upload CSV, infer types, compute reports, render controls
    """
    theme = request.args.get("theme", "light")
    sid = request.args.get("sid") or get_sid()
    session["sid"] = sid

    errors = []
    info = preview = stats_num = stats_cat = missing = None
    controls_html = None
    plots_divs: List[str] = []
    files: List[str] = []
    zip_name: Optional[str] = None
    filename = None

    # pagination & query
    q = (request.args.get("q") or "").strip()
    try:
        page = int(request.args.get("page", "1"))
        if page < 1: page = 1
    except Exception:
        page = 1
    prev_page = max(page - 1, 1)
    next_page = page + 1

    state = get_state()

    if request.method == "POST":
        try:
            upload = request.files.get("file")
            if not upload:
                raise ValueError("No file uploaded.")

            # Optional overrides
            sep = (request.form.get("sep") or "").strip() or None
            enc = (request.form.get("encoding") or "").strip() or None
            head_rows = int(request.form.get("head") or state.head_rows or 8)
            head_rows = max(head_rows, 1)

            # Robust read
            df, used_sep, used_enc = read_csv_defensively(upload, sep, enc)
            df = ensure_types(df)

            # update state
            state.filename = getattr(upload, "filename", "uploaded.csv")
            state.df = df
            state.numeric_cols, state.categorical_cols, state.date_cols = split_columns(df)
            state.head_rows = head_rows

            info = df_summary_text(df, state.filename, used_sep, used_enc)
            preview = render_table_html(df, page=1, per_page=head_rows)
            stats_num = df_numeric_stats(df)
            stats_cat = df_categorical_stats(df)
            missing = df_missing_table(df)
            filename = state.filename

            # export base reports
            files = export_reports(df, sid, state.numeric_cols, state.categorical_cols)
            state.last_reports = files
            zip_name = make_zip(files, sid)
            state.last_zip = zip_name

            controls_html = render_controls_form(state)
            flash("✅ Analysis completed.")

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            errors.append(f"{e}\n{tb}")
            flash(f"❌ Error: {e}")

    if request.method == "GET" and state.df is not None:
        df = state.df.copy()
        filename = state.filename
        if q:
            try:
                df = df.query(q)
            except Exception as qe:
                flash(f"Query error: {qe}")

        info = df_summary_text(df, filename, used_sep="?", used_encoding="?")
        preview = render_table_html(df, page=page, per_page=20)
        stats_num = df_numeric_stats(df)
        stats_cat = df_categorical_stats(df)
        missing = df_missing_table(df)
        controls_html = render_controls_form(state)
        files = [f.name for f in REPORT_DIR.glob(f"{sid}_*.*")]
        zip_name = state.last_zip

    return render_template_string(
        HTML,
        app_name=APP_NAME,
        theme=theme,
        sid=sid,
        filename=filename,
        info=info,
        preview=preview,
        stats_num=stats_num,
        stats_cat=stats_cat,
        missing=missing,
        controls=controls_html,
        plots=plots_divs,
        files=files,
        zip_name=zip_name,
        q=q,
        page=page,
        prev_page=prev_page,
        next_page=next_page,
        errors="\n".join(errors) if errors else None
    )

@app.route("/plots", methods=["POST"])
def make_plots():
    """
    Build Plotly charts based on controls.
    """
    sid = request.form.get("sid") or get_sid()
    state = get_state()
    if state.df is None:
        flash("Upload a CSV first.")
        return redirect(url_for("index"))

    df = state.df.copy()
    plots_divs: List[str] = []
    new_files: List[str] = []  # (we keep CSV/Excel from earlier exports)

    # selections
    hist_cols = request.form.getlist("hist_cols")
    box_cols = request.form.getlist("box_cols")
    violin_cols = request.form.getlist("violin_cols")
    bar_col = request.form.get("bar_col") or ""
    scatter_x = request.form.get("scatter_x") or ""
    scatter_y = request.form.get("scatter_y") or ""
    trendline = request.form.get("trendline") is not None
    date_col = request.form.get("date_col") or ""
    y_col = request.form.get("y_col") or ""
    freq = request.form.get("freq") or "M"

    # Build plots
    try:
        # Histograms
        if hist_cols:
            for c in hist_cols:
                if c in df.columns:
                    plots_divs.append(p_histogram(df, c))

        # Boxplot
        if box_cols:
            cols = [c for c in box_cols if c in df.columns]
            if cols:
                plots_divs.append(p_box(df, cols))

        # Violin
        if violin_cols:
            cols = [c for c in violin_cols if c in df.columns]
            if cols:
                plots_divs.append(p_violin(df, cols))

        # Bar (categorical)
        if bar_col and bar_col in df.columns:
            plots_divs.append(p_bar_counts(df, bar_col))

        # Scatter
        if scatter_x and scatter_y:
            if scatter_x in df.columns and scatter_y in df.columns:
                plots_divs.append(p_scatter(df, scatter_x, scatter_y, trendline=trendline))

        # Time series
        if date_col and y_col and date_col in df.columns and y_col in df.columns:
            plots_divs.append(p_timeseries(df, date_col, y_col, freq=freq))

    except Exception as e:
        flash(f"Plot error: {e}")

    flash("✅ Plots generated.")
    # re-render main template with plots
    return render_template_string(
        HTML,
        app_name=APP_NAME,
        theme="light",
        sid=sid,
        filename=state.filename,
        info=df_summary_text(df, state.filename, "?", "?"),
        preview=render_table_html(df, page=1, per_page=state.head_rows),
        stats_num=df_numeric_stats(df),
        stats_cat=df_categorical_stats(df),
        missing=df_missing_table(df),
        controls=render_controls_form(state),
        plots=plots_divs,
        files=state.last_reports,
        zip_name=state.last_zip,
        q="",
        page=1,
        prev_page=1,
        next_page=2,
        errors=None
    )

@app.route("/download/<path:filename>")
def download_file(filename):
    """Serve any file from REPORT_DIR."""
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

@app.route("/reset")
def reset_session():
    """Clear in-memory session state."""
    clear_state()
    flash("Session reset.")
    return redirect(url_for("index"))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # You can change host/port if needed
    app.run(debug=True, host="127.0.0.1", port=5000)
