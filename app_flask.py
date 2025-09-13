"""
Advanced Flask Web App: CSV Analyzer
------------------------------------
Features:
- Upload CSV
- Show dataset preview
- Show summary (info, types, missing values)
- Generate statistics (numeric + categorical)
- Show histograms, bar charts, and correlation heatmap
- Export reports (CSV + PNG)
- Modern Bootstrap interface
"""

# ----------------------
# IMPORTS
# ----------------------
import os
import io
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no GUI needed for plots)
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template_string, request, send_from_directory, redirect, url_for, flash

# ----------------------
# CONFIGURATION
# ----------------------
app = Flask(__name__)
app.secret_key = "super-secret-key"  # required for flash messages

# Directory where reports (CSV/PNG) will be saved
REPORT_DIR = Path("static/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# HTML TEMPLATE
# Using Bootstrap 5 for styling
# ----------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CSV Analyzer Web</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background-color: #f9f9f9; }
    .container { max-width: 1200px; }
    pre { background-color: #eee; padding: 1em; }
    .plot-img { max-width: 100%; margin-bottom: 20px; }
  </style>
</head>
<body>
<div class="container mt-4">
  <h1 class="mb-4">📊 CSV Analyzer (Flask Advanced)</h1>
  
  <!-- File upload form -->
  <form method="post" enctype="multipart/form-data" class="mb-4">
    <input class="form-control" type="file" name="file" accept=".csv" required>
    <button class="btn btn-primary mt-2" type="submit">Analyze</button>
  </form>

  <!-- Flash messages -->
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for msg in messages %}
        <div class="alert alert-info">{{ msg }}</div>
      {% endfor %}
    {% endif %}
  {% endwith %}
  
  <!-- Display results -->
  {% if summary %}
    <h3>Dataset Info</h3>
    <pre>{{ summary }}</pre>
  {% endif %}
  
  {% if preview %}
    <h3>Preview</h3>
    {{ preview|safe }}
  {% endif %}
  
  {% if desc_num %}
    <h3>Numeric Statistics</h3>
    {{ desc_num|safe }}
  {% endif %}
  
  {% if desc_cat %}
    <h3>Categorical Statistics</h3>
    {{ desc_cat|safe }}
  {% endif %}
  
  {% if missing %}
    <h3>Missing Values</h3>
    {{ missing|safe }}
  {% endif %}
  
  {% if plots %}
    <h3>Visualizations</h3>
    {% for p in plots %}
      <img src="data:image/png;base64,{{p}}" class="plot-img"/>
    {% endfor %}
  {% endif %}
  
  {% if files %}
    <h3>Download Reports</h3>
    <ul>
      {% for f in files %}
        <li><a href="{{ url_for('download_file', filename=f) }}" target="_blank">{{ f }}</a></li>
      {% endfor %}
    </ul>
  {% endif %}
</div>
</body>
</html>
"""

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def save_plot(fig, filename: str) -> str:
    """
    Save a matplotlib figure as PNG in REPORT_DIR and also
    return a base64 string (to embed directly in HTML).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / filename

    # Save figure to file
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)  # close figure to free memory

    # Convert image to base64 for embedding
    with open(out_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")


def analyze_csv(file) -> dict:
    """
    Perform a detailed analysis of a CSV file:
    - Summary (shape, columns, dtypes, missing values)
    - Preview (first 5 rows)
    - Numeric and categorical statistics
    - Visualizations (histograms, bar charts, correlation heatmap)
    - Save reports to disk (CSV + PNG)

    Returns:
        dict with summary text, HTML tables, plots, and list of report files.
    """
    df = pd.read_csv(file)

    # ---- SUMMARY ----
    summary = []
    summary.append(f"File loaded: {getattr(file, 'filename', 'uploaded.csv')}")
    summary.append(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    summary.append("Columns: " + ", ".join(df.columns))
    summary.append("\nColumn types:\n" + str(df.dtypes))
    summary_text = "\n".join(summary)

    # ---- PREVIEW ----
    preview_html = df.head().to_html(classes="table table-striped table-sm", index=False)

    # ---- STATISTICS ----
    desc_num_html = ""
    desc_cat_html = ""
    missing_html = ""

    # Numeric stats
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc_num_html = num_df.describe().T.to_html(classes="table table-bordered table-sm")

    # Categorical stats
    cat_df = df.select_dtypes(exclude=[np.number])
    if not cat_df.empty:
        desc_cat_html = cat_df.describe().T.to_html(classes="table table-bordered table-sm")

    # Missing values
    missing = df.isna().sum()
    if missing.any():
        missing_html = missing.to_frame("missing_count").to_html(classes="table table-bordered table-sm")

    # ---- VISUALIZATIONS ----
    plots = []

    # Histograms for numeric columns
    for col in num_df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram — {col}")
        plots.append(save_plot(fig, f"hist_{col}.png"))

    # Bar charts for categorical columns
    for col in cat_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].value_counts().plot(kind="bar", ax=ax, color="orange")
        ax.set_title(f"Bar Chart — {col}")
        plots.append(save_plot(fig, f"bar_{col}.png"))

    # Correlation heatmap (if at least 2 numeric columns)
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        plots.append(save_plot(fig, "heatmap_corr.png"))

        # Save correlation matrix as CSV
        corr.to_csv(REPORT_DIR / "correlation.csv", index=True)

    # ---- EXPORT REPORTS ----
    # Use timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    head_file = f"head_{timestamp}.csv"
    num_file = f"numeric_stats_{timestamp}.csv"
    cat_file = f"categorical_stats_{timestamp}.csv"
    miss_file = f"missing_{timestamp}.csv"

    # Save to CSV
    df.head().to_csv(REPORT_DIR / head_file, index=False)
    if not num_df.empty:
        num_df.describe().T.to_csv(REPORT_DIR / num_file)
    if not cat_df.empty:
        cat_df.describe().T.to_csv(REPORT_DIR / cat_file)
    if missing.any():
        missing.to_csv(REPORT_DIR / miss_file)

    # Collect list of generated files
    files = [head_file]
    if not num_df.empty: files.append(num_file)
    if not cat_df.empty: files.append(cat_file)
    if missing.any(): files.append(miss_file)
    files += [f.name for f in REPORT_DIR.glob("*.png")]

    return {
        "summary": summary_text,
        "preview": preview_html,
        "desc_num": desc_num_html,
        "desc_cat": desc_cat_html,
        "missing": missing_html,
        "plots": plots,
        "files": files,
    }

# ----------------------
# ROUTES
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Homepage:
    - GET: show upload form
    - POST: process uploaded CSV and display analysis
    """
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("⚠️ No file uploaded")
            return redirect(url_for("index"))
        try:
            result = analyze_csv(file)
            flash("✅ Analysis completed successfully")
            return render_template_string(HTML, **result)
        except Exception as e:
            flash(f"❌ Error: {e}")
            return redirect(url_for("index"))
    # On GET: render empty page with form
    return render_template_string(HTML, summary=None, preview=None,
                                  desc_num=None, desc_cat=None,
                                  missing=None, plots=None, files=None)


@app.route("/download/<path:filename>")
def download_file(filename):
    """Allow downloading of generated report files (CSV/PNG)."""
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

# ----------------------
# ENTRY POINT
# ----------------------
if __name__ == "__main__":
    # Run Flask app in debug mode
    app.run(debug=True, port=5000)
