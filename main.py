import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------- Utility functions --------
def smart_read_csv(path: Path, sep: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    """
    Try to read a CSV file intelligently.
    - If a separator is provided, use it directly.
    - Otherwise, try a list of common separators: ',', ';', tab, '|'.
    - Returns the DataFrame once successfully parsed.
    """
    if sep:
        return pd.read_csv(path, sep=sep, encoding=encoding)

    # Try different separators until one works
    for cand in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=cand, encoding=encoding)
            # Accept if it produces at least one column
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue

    # Fallback: try with default pandas settings
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    """
    Ensure that a directory exists.
    If not, it is created (recursively).
    """
    p.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    """
    Save a DataFrame as a CSV file inside a target directory.
    - df: DataFrame to save
    - out_dir: target folder (created if missing)
    - name: output file name
    Returns the path of the saved file.
    """
    ensure_dir(out_dir)
    out_path = out_dir / name
    df.to_csv(out_path, index=False)
    print(f"💾 Saved: {out_path}")
    return out_path


# -------- Main analysis logic --------
def analyze_csv(csv_path: Path, sep: str | None, encoding: str | None, report: bool, head_rows: int) -> None:
    """
    Perform analysis on a CSV file:
    - Show basic info (rows, columns, column names, dtypes)
    - Show first rows
    - Show descriptive statistics for numeric columns
    - Show missing values
    - Show correlation matrix for numeric columns
    - Optionally, generate a full report (CSV files + histogram images)
    """
    print(f"📂 Loading: {csv_path}")
    df = smart_read_csv(csv_path, sep=sep, encoding=encoding)

    # --- Basic info ---
    print("\n=== BASIC INFO ===")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    # --- Data types ---
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)

    # --- First rows ---
    print(f"\n=== FIRST {head_rows} ROWS ===")
    print(df.head(head_rows))

    # --- Descriptive statistics for numeric columns ---
    print("\n=== NUMERIC DESCRIPTION ===")
    with pd.option_context("display.max_columns", None):
        print(df.describe(include=[np.number]).transpose())

    # --- Missing values ---
    print("\n=== MISSING VALUES PER COLUMN ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing)

    # --- Correlation matrix (numeric only) ---
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True).round(3)
        print("\n=== CORRELATION MATRIX ===")
        print(corr)
    else:
        corr = pd.DataFrame()
        print("\n(No correlation: less than 2 numeric columns)")

    # --- Save report (optional) ---
    if report:
        out = csv_path.parent / "reports"
        ensure_dir(out)

        # Save head
        save_df(df.head(head_rows), out, "head.csv")

        # Save numeric description
        save_df(
            df.describe(include=[np.number]).transpose().reset_index(names="column"),
            out,
            "describe_numeric.csv"
        )

        # Save missing values
        save_df(
            missing.reset_index().rename(columns={"index": "column", 0: "missing"}),
            out,
            "missing.csv"
        )

        # Save correlation matrix if available
        if not corr.empty:
            save_df(corr.reset_index(), out, "correlation.csv")

        # Save histograms for each numeric column
        for col in num_df.columns:
            plt.figure()
            df[col].plot(kind="hist", bins=30, title=f"Histogram — {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            img_path = out / f"hist_{col}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            print(f"🖼️ Saved plot: {img_path}")

        print("\n✅ Report generated in the 'reports/' folder")


# -------- Command-line interface --------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the command-line interface.
    Options:
    - --file / -f: path to CSV (default: vendite.csv)
    - --sep: CSV separator (default: auto-detect)
    - --encoding: encoding (default: None, usually utf-8)
    - --head: number of rows to display (default: 5)
    - --report: flag to enable report saving
    """
    p = argparse.ArgumentParser(description="CSV Analyzer (pandas)")
    p.add_argument("--file", "-f", help="Path to CSV file", default="vendite.csv")
    p.add_argument("--sep", help="Separator (default: auto). E.g.: ',', ';', '\\t', '|'")
    p.add_argument("--encoding", help="Encoding (e.g.: 'utf-8', 'latin-1')", default=None)
    p.add_argument("--head", type=int, default=5, help="How many rows to show (default: 5)")
    p.add_argument("--report", action="store_true", help="Save report and plots in 'reports/'")
    return p


def main():
    """
    Entry point:
    - Parse arguments
    - If no arguments are given, use defaults: vendite.csv + generate report
    - Run the analysis
    """
    parser = build_parser()

    # If no arguments are provided, add defaults
    if len(sys.argv) == 1:
        sys.argv.extend(["--file", "vendite.csv", "--report"])

    args = parser.parse_args()
    csv_path = Path(args.file)

    # Check file existence
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    # Run analysis
    analyze_csv(csv_path, sep=args.sep, encoding=args.encoding, report=args.report, head_rows=args.head)


if __name__ == "__main__":
    main()
